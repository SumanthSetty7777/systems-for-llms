#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>

namespace {

__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warp_max(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return __shfl_sync(0xffffffff, value, 0);
}

// One CTA shares a KV tile among the query heads belonging to one KV head.
// Splits add independent CTAs along context length when B*Hkv is small.
template <typename scalar_t, int HeadDim>
__global__ void partial_attention(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int32_t* __restrict__ lengths,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_output,
    scalar_t* __restrict__ output,
    int query_heads, int kv_heads, int capacity, int groups, int splits,
    float scale) {
    constexpr int Tile = 32;
    constexpr int Elements = HeadDim / 32;
    __shared__ scalar_t tile_k[Tile * HeadDim];
    __shared__ scalar_t tile_v[Tile * HeadDim];
    const int batch = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int split = blockIdx.z;
    const int lane = threadIdx.x % 32;
    const int group = threadIdx.x / 32;
    const int head = kv_head * groups + group;
    const int valid = max(0, min(lengths[batch], capacity));
    const int chunk = (capacity + splits - 1) / splits;
    const int begin = split * chunk;
    const int end = min(valid, begin + chunk);
    const int64_t cache_base = (int64_t(batch) * kv_heads + kv_head) * capacity * HeadDim;
    const int64_t query_base = (int64_t(batch) * query_heads + head) * HeadDim;
    float query[Elements];
    float numerator[Elements];
    #pragma unroll
    for (int item = 0; item < Elements; ++item) {
        query[item] = group < groups ? float(q[query_base + lane + item * 32]) : 0.0f;
        numerator[item] = 0.0f;
    }
    float maximum = -INFINITY;
    float normalizer = 0.0f;

    for (int tile_start = begin; tile_start < end; tile_start += Tile) {
        const int count = min(Tile, end - tile_start);
        for (int index = threadIdx.x; index < count * HeadDim; index += blockDim.x) {
            tile_k[index] = k[cache_base + int64_t(tile_start) * HeadDim + index];
            tile_v[index] = v[cache_base + int64_t(tile_start) * HeadDim + index];
        }
        __syncthreads();
        if (group < groups) {
            for (int token = 0; token < count; ++token) {
                float dot = 0.0f;
                #pragma unroll
                for (int item = 0; item < Elements; ++item)
                    dot += query[item] * float(tile_k[token * HeadDim + lane + item * 32]);
                const float score = warp_sum(dot) * scale;
                const float next_maximum = fmaxf(maximum, score);
                const float old_weight = expf(maximum - next_maximum);
                const float new_weight = expf(score - next_maximum);
                normalizer = normalizer * old_weight + new_weight;
                #pragma unroll
                for (int item = 0; item < Elements; ++item)
                    numerator[item] = numerator[item] * old_weight +
                        new_weight * float(tile_v[token * HeadDim + lane + item * 32]);
                maximum = next_maximum;
            }
        }
        __syncthreads();
    }
    if (group >= groups) return;
    if (splits == 1) {
        #pragma unroll
        for (int item = 0; item < Elements; ++item)
            output[query_base + lane + item * 32] =
                scalar_t(normalizer > 0 ? numerator[item] / normalizer : 0.0f);
        return;
    }
    const int64_t row = (int64_t(batch) * query_heads + head) * splits + split;
    if (lane == 0) {
        partial_max[row] = maximum;
        partial_sum[row] = normalizer;
    }
    #pragma unroll
    for (int item = 0; item < Elements; ++item)
        partial_output[row * HeadDim + lane + item * 32] = numerator[item];
}

template <typename scalar_t, int HeadDim>
__global__ void merge_partials(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_output,
    scalar_t* __restrict__ output, int splits) {
    __shared__ float warp_values[4];
    __shared__ float weights[128];
    __shared__ float row_maximum;
    __shared__ float denominator;
    const int thread = threadIdx.x;
    const int lane = thread % 32;
    const int warp = thread / 32;
    const int64_t row = int64_t(blockIdx.x) * splits;
    float maximum = thread < splits ? partial_max[row + thread] : -INFINITY;
    maximum = warp_max(maximum);
    if (lane == 0) warp_values[warp] = maximum;
    __syncthreads();
    if (warp == 0) {
        maximum = warp_max(lane < 4 ? warp_values[lane] : -INFINITY);
        if (lane == 0) row_maximum = maximum;
    }
    __syncthreads();
    float term = 0.0f;
    if (thread < splits) {
        const float maximum_for_split = partial_max[row + thread];
        const float weight = isfinite(row_maximum) && isfinite(maximum_for_split)
            ? expf(maximum_for_split - row_maximum) : 0.0f;
        weights[thread] = weight;
        term = weight * partial_sum[row + thread];
    }
    term = warp_sum(term);
    if (lane == 0) warp_values[warp] = term;
    __syncthreads();
    if (warp == 0) {
        term = warp_sum(lane < 4 ? warp_values[lane] : 0.0f);
        if (lane == 0) denominator = term;
    }
    __syncthreads();
    if (thread < HeadDim) {
        float value = 0.0f;
        for (int split = 0; split < splits; ++split)
            value += weights[split] * partial_output[(row + split) * HeadDim + thread];
        output[int64_t(blockIdx.x) * HeadDim + thread] =
            scalar_t(denominator > 0 ? value / denominator : 0.0f);
    }
}

template <typename scalar_t, int HeadDim>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& lengths, at::Tensor& maximum, at::Tensor& normalizer,
            at::Tensor& numerator, at::Tensor& output, int splits, float scale,
            cudaStream_t stream) {
    const int batch = q.size(0), query_heads = q.size(1), kv_heads = k.size(1);
    const int groups = query_heads / kv_heads;
    partial_attention<scalar_t, HeadDim><<<dim3(batch, kv_heads, splits),
        32 * std::max(4, groups), 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        lengths.data_ptr<int32_t>(), maximum.data_ptr<float>(), normalizer.data_ptr<float>(),
        numerator.data_ptr<float>(), output.data_ptr<scalar_t>(), query_heads,
        kv_heads, k.size(2), groups, splits, scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (splits > 1) {
        merge_partials<scalar_t, HeadDim><<<batch * query_heads, 128, 0, stream>>>(
            maximum.data_ptr<float>(), normalizer.data_ptr<float>(),
            numerator.data_ptr<float>(), output.data_ptr<scalar_t>(), splits);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

void run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor lengths,
         at::Tensor maximum, at::Tensor normalizer, at::Tensor numerator,
         at::Tensor output, int64_t splits, double scale) {
    TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.sizes() == k.sizes(), "invalid Q/K/V shapes");
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "Q must be contiguous CUDA storage");
    for (const auto& tensor : {k, v, lengths, maximum, normalizer, numerator, output}) {
        TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                    "all tensors must be contiguous on the Q CUDA device");
    }
    const int64_t batch = q.size(0), heads = q.size(1), dim = q.size(2);
    TORCH_CHECK(batch > 0 && heads > 0 && (dim == 64 || dim == 128), "unsupported Q shape");
    TORCH_CHECK(k.size(0) == batch && k.size(3) == dim && k.size(1) > 0 && k.size(2) > 0,
                "invalid KV shape");
    TORCH_CHECK(heads % k.size(1) == 0, "query heads must be divisible by KV heads");
    const auto groups = heads / k.size(1);
    TORCH_CHECK(groups == 1 || groups == 2 || groups == 4 || groups == 8, "unsupported GQA ratio");
    TORCH_CHECK(splits >= 1 && splits <= 128, "split count must be 1 through 128");
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "invalid scale");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "Q must be FP16 or BF16");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type() &&
                output.scalar_type() == q.scalar_type() && output.sizes() == q.sizes(),
                "Q/K/V/output dtype or output shape mismatch");
    TORCH_CHECK(lengths.scalar_type() == at::kInt && lengths.sizes() == at::IntArrayRef({batch}),
                "lengths must be int32[B]");
    TORCH_CHECK(maximum.scalar_type() == at::kFloat && normalizer.scalar_type() == at::kFloat &&
                numerator.scalar_type() == at::kFloat, "scratch must be FP32");
    TORCH_CHECK(maximum.sizes() == at::IntArrayRef({batch, heads, splits}) &&
                normalizer.sizes() == maximum.sizes() &&
                numerator.sizes() == at::IntArrayRef({batch, heads, splits, dim}),
                "scratch shape mismatch");
    TORCH_CHECK(batch <= 65535 && k.size(1) <= 65535 && k.size(2) <= INT_MAX - 127 &&
                heads <= INT_MAX && batch * heads <= INT_MAX, "shape exceeds kernel index limits");
    c10::cuda::CUDAGuard device_guard(q.device());
    const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    if (q.scalar_type() == at::kHalf) {
        if (dim == 64) launch<at::Half, 64>(q, k, v, lengths, maximum, normalizer, numerator, output, splits, scale, stream);
        else launch<at::Half, 128>(q, k, v, lengths, maximum, normalizer, numerator, output, splits, scale, stream);
    } else {
        if (dim == 64) launch<at::BFloat16, 64>(q, k, v, lengths, maximum, normalizer, numerator, output, splits, scale, stream);
        else launch<at::BFloat16, 128>(q, k, v, lengths, maximum, normalizer, numerator, output, splits, scale, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, "Split-KV grouped-query decode into prepared storage");
}
