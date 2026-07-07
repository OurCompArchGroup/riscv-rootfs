#include "ame.h"
#include "mem.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TILE_M = 64,
    TILE_N = 64,
    TILE_K = 64,
};

_Static_assert(AME_TILE_M == TILE_M, "AME_TILE_M must match FPGA tile M");
_Static_assert(AME_TILE_N == TILE_N, "AME_TILE_N must match FPGA tile N");
_Static_assert(AME_TILE_K == TILE_K, "AME_TILE_K must match FPGA tile K");

static int env_int(const char * name, int fallback) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return atoi(value);
}

static int env_flag(const char * name, int fallback) {
    return env_int(name, fallback) != 0;
}

static void fill_a_panel(int8_t * a_panel, int panel_kb, int m_tiles) {
    const int tile_a_elems = TILE_M * TILE_K;
    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kb = 0; kb < panel_kb; ++kb) {
            int8_t * tile = a_panel + ((size_t) mt * panel_kb + kb) * tile_a_elems;
            for (int i = 0; i < tile_a_elems; ++i) {
                tile[i] = (int8_t) (((i + 3 * mt + kb) % 17) - 8);
            }
        }
    }
}

static int should_log_progress(int index, int last, int interval) {
    return interval > 0 && (index == 0 || index == last || (index % interval) == 0);
}

static int64_t env_i64(const char * name, int64_t fallback) {
    const char * value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return strtoll(value, NULL, 0);
}

static void full_fence(void) {
    asm volatile("fence rw,rw" ::: "memory");
}

static uint64_t read_cycle(void) {
    uint64_t value;
    asm volatile("rdcycle %0" : "=r"(value));
    return value;
}

static void fill_b_panel(int8_t * b, int panel_kb) {
    const int tile_b_elems = TILE_N * TILE_K;
    for (int kb = 0; kb < panel_kb; ++kb) {
        int8_t * tile = b + (size_t) kb * tile_b_elems;
        for (int i = 0; i < tile_b_elems; ++i) {
            tile[i] = (int8_t) (((i + kb) % 11) - 5);
        }
    }
}

static void fill_scales(float * x_scales, float * y_scales, int panel_kb, int m_tiles) {
    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kb = 0; kb < panel_kb; ++kb) {
            x_scales[(size_t) mt * panel_kb + kb] = 1.0f / (float) (1 + ((mt + kb) & 7));
        }
    }
    for (int kb = 0; kb < panel_kb; ++kb) {
        for (int j = 0; j < TILE_N; ++j) {
            y_scales[(size_t) kb * TILE_N + j] = 1.0f / (float) (1 + ((j + kb) & 15));
        }
    }
}

static uint32_t checksum_c(const int32_t * c) {
    uint32_t acc = 2166136261u;
    for (int i = 0; i < TILE_M * TILE_N; ++i) {
        acc ^= (uint32_t) c[i];
        acc *= 16777619u;
    }
    return acc;
}

static uint32_t checksum_f32(const float * data, size_t count) {
    uint32_t acc = 2166136261u;
    for (size_t i = 0; i < count; ++i) {
        union {
            float f;
            uint32_t u;
        } v;
        v.f = data[i];
        acc ^= v.u;
        acc *= 16777619u;
    }
    return acc;
}

static void accumulate_scaled_row(float * acc, const int32_t * c, float x_scale, const float * y_scales) {
    for (int j = 0; j < TILE_N; ++j) {
        acc[j] += (float) c[j] * x_scale * y_scales[j];
    }
}

static void store_acc_tile(float * out, const float * acc, int m_tiles, int mt) {
    for (int i = 0; i < TILE_M; ++i) {
        memcpy(out + ((size_t) mt * TILE_M + i) * TILE_N,
               acc + (size_t) i * TILE_N,
               TILE_N * sizeof(float));
    }
    (void) m_tiles;
}

int main(void) {
    const int panel_kb = env_int("AME_PANEL_PROBE_KB", 128);
    const int passes   = env_int("AME_PANEL_PROBE_PASSES", 1);
    const int m_tiles  = env_int("AME_PANEL_PROBE_M_TILES", 16);
    const int start_mtile = env_int("AME_PANEL_PROBE_START_MTILE", 0);
    const int reuse_mtile = env_flag("AME_PANEL_PROBE_REUSE_MTILE", 0);
    const int start_kb = env_int("AME_PANEL_PROBE_START_KB", 0);
    const int step_kb  = env_int("AME_PANEL_PROBE_STEP_KB", 1);
    const int packed_loop = env_flag("AME_PANEL_PROBE_PACKED_LOOP", 0);
    const int do_scale = env_flag("AME_PANEL_PROBE_SCALE", packed_loop);
    const int do_store = env_flag("AME_PANEL_PROBE_STORE", packed_loop);
    const int do_pack_a = env_flag("AME_PANEL_PROBE_PACK_A", packed_loop);
    const int log_each = env_flag("AME_PANEL_PROBE_LOG_EACH", packed_loop ? 0 : 1);
    const int progress_interval = env_int("AME_PANEL_PROBE_PROGRESS_INTERVAL", log_each ? 1 : 100);
    const int iter_progress = env_flag("AME_PANEL_PROBE_ITER_PROGRESS", 0);
    const int c_buffers = env_int("AME_PANEL_PROBE_C_BUFFERS", 1);
    const int zero_c = env_flag("AME_PANEL_PROBE_ZERO_C", 1);
    const int fence_before = env_flag("AME_PANEL_PROBE_FENCE_BEFORE", 0);
    const int fence_after = env_flag("AME_PANEL_PROBE_FENCE_AFTER", 0);
    const int legacy_hot_loop = env_flag("AME_PANEL_PROBE_LEGACY_HOT_LOOP", 0);
    const int64_t detail_start = env_i64("AME_PANEL_PROBE_DETAIL_START", -1);
    const int64_t detail_end = env_i64("AME_PANEL_PROBE_DETAIL_END", -1);
    const int64_t stop_after_total = env_i64("AME_PANEL_PROBE_STOP_AFTER_TOTAL", -1);
    int limit_kb       = env_int("AME_PANEL_PROBE_LIMIT_KB", panel_kb);

    if (panel_kb <= 0 || passes <= 0 || m_tiles <= 0 || start_mtile < 0 || start_kb < 0 || step_kb <= 0 || c_buffers <= 0) {
        fprintf(stderr, "[ame-panel-probe] invalid arguments\n");
        return 2;
    }
    if (limit_kb > panel_kb) {
        limit_kb = panel_kb;
    }
    if (start_kb >= limit_kb) {
        fprintf(stderr, "[ame-panel-probe] empty scan range\n");
        return 2;
    }

    const size_t tile_a_bytes = (size_t) TILE_M * TILE_K * sizeof(int8_t);
    const size_t tile_b_bytes = (size_t) TILE_N * TILE_K * sizeof(int8_t);
    const size_t tile_c_bytes = (size_t) TILE_M * TILE_N * sizeof(int32_t);
    const size_t panel_bytes = (size_t) panel_kb * tile_b_bytes;
    const int a_mtiles = packed_loop ? (reuse_mtile ? start_mtile + 1 : start_mtile + m_tiles) : 1;
    const size_t a_panel_bytes = (size_t) panel_kb * a_mtiles * tile_a_bytes;
    const size_t c_pool_bytes = (size_t) c_buffers * tile_c_bytes;
    const size_t acc_bytes = (size_t) TILE_M * TILE_N * sizeof(float);
    const size_t out_bytes = (size_t) m_tiles * TILE_M * TILE_N * sizeof(float);

    printf("[ame-panel-probe] tile M=%d N=%d K=%d\n", TILE_M, TILE_N, TILE_K);
    printf("[ame-panel-probe] panel_kb=%d panel_bytes=%zu passes=%d m_tiles=%d start_mtile=%d reuse_mtile=%d start=%d limit=%d step=%d packed_loop=%d pack_a=%d scale=%d store=%d log_each=%d progress_interval=%d iter_progress=%d c_buffers=%d zero_c=%d fence_before=%d fence_after=%d legacy_hot_loop=%d detail=[%lld,%lld] stop_after_total=%lld\n",
           panel_kb, panel_bytes, passes, m_tiles, start_mtile, reuse_mtile, start_kb, limit_kb, step_kb,
           packed_loop, do_pack_a, do_scale, do_store, log_each, progress_interval,
           iter_progress, c_buffers, zero_c, fence_before, fence_after, legacy_hot_loop,
           (long long) detail_start, (long long) detail_end, (long long) stop_after_total);

    int8_t  * a = (int8_t *) my_malloc(packed_loop ? a_panel_bytes : tile_a_bytes);
    int8_t  * tile_a = (int8_t *) my_malloc(tile_a_bytes);
    int8_t  * b = (int8_t *) my_malloc(panel_bytes);
    int32_t * c_pool = (int32_t *) my_malloc(c_pool_bytes);
    float * acc = (float *) my_malloc(acc_bytes);
    float * out = do_store ? (float *) my_malloc(out_bytes) : NULL;
    float * x_scales = (float *) my_malloc((size_t) panel_kb * a_mtiles * sizeof(float));
    float * y_scales = (float *) my_malloc((size_t) panel_kb * TILE_N * sizeof(float));
    if (a == NULL || tile_a == NULL || b == NULL || c_pool == NULL || acc == NULL ||
        (do_store && out == NULL) || x_scales == NULL || y_scales == NULL) {
        fprintf(stderr, "[ame-panel-probe] allocation failed a=%p tile_a=%p b=%p c_pool=%p acc=%p out=%p xs=%p ys=%p\n",
                (void *) a, (void *) tile_a, (void *) b, (void *) c_pool,
                (void *) acc, (void *) out, (void *) x_scales, (void *) y_scales);
        return 3;
    }

    printf("[ame-panel-probe] a_panel=%p tile_a=%p b_panel=%p c_pool=%p acc=%p out=%p tile_b_bytes=%zu tile_c_bytes=%zu\n",
           (void *) a, (void *) tile_a, (void *) b, (void *) c_pool, (void *) acc, (void *) out, tile_b_bytes, tile_c_bytes);
    ggml_ame_init();
    fill_a_panel(a, packed_loop ? panel_kb : 1, packed_loop ? a_mtiles : 1);
    if (!packed_loop) {
        memcpy(tile_a, a, tile_a_bytes);
    }
    fill_b_panel(b, panel_kb);
    fill_scales(x_scales, y_scales, panel_kb, m_tiles);
    memset(c_pool, 0, c_pool_bytes);

    int64_t total = 0;

    if (packed_loop && legacy_hot_loop) {
        int32_t * c = c_pool;
        for (int pass = 0; pass < passes; ++pass) {
            printf("[ame-panel-probe] pass %d/%d\n", pass + 1, passes);
            for (int iter = 0; iter < m_tiles; ++iter) {
                const int mt = reuse_mtile ? start_mtile : start_mtile + iter;
                const int detail_log = detail_start >= 0 && total >= detail_start && (detail_end < 0 || total <= detail_end);
                const int log_progress = detail_log || should_log_progress(iter, m_tiles - 1, progress_interval);
                const int kb_progress = detail_log || (log_progress && !iter_progress);
                memset(acc, 0, acc_bytes);
                if (log_progress) {
                    printf("[ame-panel-probe] iter=%d mt=%d begin\n", iter, mt);
                    fflush(stdout);
                }
                for (int kb = start_kb; kb < limit_kb; kb += step_kb) {
                    int8_t * b_tile = b + (size_t) kb * tile_b_bytes;
                    const int8_t * a_src = a + ((size_t) mt * panel_kb + kb) * tile_a_bytes;
                    if (do_pack_a) {
                        memcpy(tile_a, a_src, tile_a_bytes);
                    } else {
                        tile_a = (int8_t *) a_src;
                    }
                    if (zero_c) {
                        memset(c, 0, tile_c_bytes);
                    }
                    if (kb_progress) {
                        printf("[ame-panel-probe] before total=%lld pass=%d iter=%d mt=%d kb=%d b_offset=%zu cyc=%llu\n",
                               (long long) total, pass + 1, iter, mt, kb, (size_t) kb * tile_b_bytes,
                               (unsigned long long) read_cycle());
                        fflush(stdout);
                    }
                    ggml_ame_gemm_tile_i8_i32_bT(tile_a, b_tile, c);
                    if (detail_log) {
                        printf("[ame-panel-probe] after  total=%lld pass=%d iter=%d mt=%d kb=%d cyc=%llu csum=%08x\n",
                               (long long) total, pass + 1, iter, mt, kb,
                               (unsigned long long) read_cycle(), checksum_c(c));
                        fflush(stdout);
                    }
                    if (do_scale) {
                        for (int i = 0; i < TILE_M; ++i) {
                            const float xs = x_scales[(size_t) mt * panel_kb + kb];
                            accumulate_scaled_row(&acc[(size_t) i * TILE_N],
                                                  &c[(size_t) i * TILE_N],
                                                  xs,
                                                  &y_scales[(size_t) kb * TILE_N]);
                        }
                    }
                    if (log_each) {
                        printf("[ame-panel-probe] after  mt=%d kb=%d csum=%08x acc=%08x\n",
                               mt, kb, checksum_c(c), do_scale ? checksum_f32(acc, TILE_M * TILE_N) : 0);
                        fflush(stdout);
                    }
                    if (stop_after_total >= 0 && total >= stop_after_total) {
                        printf("[ame-panel-probe] stop_after_total reached total=%lld\n", (long long) total);
                        return 0;
                    }
                    ++total;
                }
                if (do_store) {
                    store_acc_tile(out, acc, m_tiles, iter);
                    if (log_each) {
                        printf("[ame-panel-probe] mt=%d stored out_checksum=%08x\n",
                               mt, checksum_f32(out, (size_t) m_tiles * TILE_M * TILE_N));
                    } else {
                        printf("[ame-panel-probe] mt=%d stored\n", mt);
                    }
                    fflush(stdout);
                }
            }
        }

        printf("[ame-panel-probe] done\n");
        return 0;
    }

    for (int pass = 0; pass < passes; ++pass) {
        printf("[ame-panel-probe] pass %d/%d\n", pass + 1, passes);
        if (packed_loop) {
            for (int iter = 0; iter < m_tiles; ++iter) {
                const int mt = reuse_mtile ? start_mtile : start_mtile + iter;
                const int detail_log = detail_start >= 0 && total >= detail_start && (detail_end < 0 || total <= detail_end);
                const int log_progress = detail_log || should_log_progress(iter, m_tiles - 1, progress_interval);
                const int kb_progress = detail_log || (log_progress && !iter_progress);
                memset(acc, 0, acc_bytes);
                if (log_progress) {
                    printf("[ame-panel-probe] total=%lld pass=%d iter=%d mt=%d begin\n",
                           (long long) total, pass + 1, iter, mt);
                    fflush(stdout);
                }
                for (int kb = start_kb; kb < limit_kb; kb += step_kb) {
                    if (stop_after_total >= 0 && total > stop_after_total) {
                        printf("[ame-panel-probe] stop_after_total reached total=%lld\n", (long long) total);
                        return 0;
                    }
                    int8_t * b_tile = b + (size_t) kb * tile_b_bytes;
                    const int8_t * a_src = a + ((size_t) mt * panel_kb + kb) * tile_a_bytes;
                    int32_t * c = c_pool + (size_t) (total % c_buffers) * TILE_M * TILE_N;
                    if (do_pack_a) {
                        memcpy(tile_a, a_src, tile_a_bytes);
                    } else {
                        tile_a = (int8_t *) a_src;
                    }
                    if (zero_c) {
                        memset(c, 0, tile_c_bytes);
                    }
                    if (fence_before) {
                        full_fence();
                    }
                    if (kb_progress) {
                        printf("[ame-panel-probe] before total=%lld pass=%d iter=%d mt=%d kb=%d b_offset=%zu c_slot=%lld c=%p\n",
                               (long long) total, pass + 1, iter, mt, kb, (size_t) kb * tile_b_bytes,
                               (long long) (total % c_buffers), (void *) c);
                        fflush(stdout);
                    }
                    ggml_ame_gemm_tile_i8_i32_bT(tile_a, b_tile, c);
                    if (fence_after) {
                        full_fence();
                    }
                    if (do_scale) {
                        for (int i = 0; i < TILE_M; ++i) {
                            const float xs = x_scales[(size_t) mt * panel_kb + kb];
                            accumulate_scaled_row(&acc[(size_t) i * TILE_N],
                                                  &c[(size_t) i * TILE_N],
                                                  xs,
                                                  &y_scales[(size_t) kb * TILE_N]);
                        }
                    }
                    if (log_each) {
                        printf("[ame-panel-probe] after  mt=%d kb=%d csum=%08x acc=%08x\n",
                               mt, kb, checksum_c(c), do_scale ? checksum_f32(acc, TILE_M * TILE_N) : 0);
                        fflush(stdout);
                    }
                    ++total;
                }
                if (do_store) {
                    store_acc_tile(out, acc, m_tiles, iter);
                    if (log_each) {
                        printf("[ame-panel-probe] mt=%d stored out_checksum=%08x\n",
                               mt, checksum_f32(out, (size_t) m_tiles * TILE_M * TILE_N));
                    } else {
                        printf("[ame-panel-probe] mt=%d stored\n", mt);
                    }
                    fflush(stdout);
                }
            }
        } else {
            for (int kb = start_kb; kb < limit_kb; kb += step_kb) {
                int8_t * b_tile = b + (size_t) kb * tile_b_bytes;
                for (int mt = 0; mt < m_tiles; ++mt) {
                    int32_t * c = c_pool + (size_t) (total % c_buffers) * TILE_M * TILE_N;
                    if (zero_c) {
                        memset(c, 0, tile_c_bytes);
                    }
                    if (fence_before) {
                        full_fence();
                    }
                    if (log_each) {
                        printf("[ame-panel-probe] before total=%lld kb=%d mt=%d b_offset=%zu b_ptr=%p c_slot=%lld c=%p\n",
                               (long long) total, kb, mt, (size_t) kb * tile_b_bytes, (void *) b_tile,
                               (long long) (total % c_buffers), (void *) c);
                        fflush(stdout);
                    }
                    ggml_ame_gemm_tile_i8_i32_bT(tile_a, b_tile, c);
                    if (fence_after) {
                        full_fence();
                    }
                    if (log_each) {
                        printf("[ame-panel-probe] after  kb=%d mt=%d checksum=%08x\n",
                               kb, mt, checksum_c(c));
                        fflush(stdout);
                    }
                    ++total;
                }
            }
        }
    }

    printf("[ame-panel-probe] done\n");
    return 0;
}
