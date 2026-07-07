#include <stdint.h>
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>

static sigjmp_buf csr_jmp;

static void handle_sigill(int signo) {
    (void) signo;
    siglongjmp(csr_jmp, 1);
}

static int read_ame_csrs(uint64_t * tlenb, uint64_t * trlenb, uint64_t * alenb) {
#if defined(__riscv)
    struct sigaction old_action;
    struct sigaction action;

    action.sa_handler = handle_sigill;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    if (sigaction(SIGILL, &action, &old_action) != 0) {
        return 0;
    }

    if (sigsetjmp(csr_jmp, 1) != 0) {
        sigaction(SIGILL, &old_action, NULL);
        return 0;
    }

    __asm__ volatile("csrr %0, 0xcc1" : "=r"(*tlenb));
    __asm__ volatile("csrr %0, 0xcc2" : "=r"(*trlenb));
    __asm__ volatile("csrr %0, 0xcc3" : "=r"(*alenb));

    sigaction(SIGILL, &old_action, NULL);
    return 1;
#else
    (void) tlenb;
    (void) trlenb;
    (void) alenb;
    return 0;
#endif
}

static const char * int8_capacity_tag(uint64_t int8_m_max, uint64_t int8_k_max, uint64_t int32_n_max) {
    if (int8_m_max >= 128 && int8_k_max >= 64 && int32_n_max >= 128) {
        return "m128n128k64.i8i32";
    }
    if (int8_m_max >= 64 && int8_k_max >= 64 && int32_n_max >= 64) {
        return "m64n64k64.i8i32";
    }
    return "none";
}

int main(void) {
    uint64_t tlenb = 0;
    uint64_t trlenb = 0;
    uint64_t alenb = 0;

    if (!read_ame_csrs(&tlenb, &trlenb, &alenb)) {
        printf("[xsai-ame] unavailable: failed to read tlenb/trlenb/alenb\n");
        return 0;
    }

    if (tlenb == 0 || trlenb == 0 || alenb == 0 ||
            tlenb == UINT64_MAX || trlenb == UINT64_MAX || alenb == UINT64_MAX ||
            tlenb % trlenb != 0) {
        printf("[xsai-ame] unavailable: tlenb=%llu trlenb=%llu alenb=%llu\n",
            (unsigned long long) tlenb,
            (unsigned long long) trlenb,
            (unsigned long long) alenb);
        return 0;
    }

    const uint64_t rows = tlenb / trlenb;
    if (rows == 0 || alenb % rows != 0 || rows > UINT64_MAX / rows || alenb > UINT64_MAX / 8) {
        printf("[xsai-ame] unavailable: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu\n",
            (unsigned long long) tlenb,
            (unsigned long long) trlenb,
            (unsigned long long) alenb,
            (unsigned long long) rows);
        return 0;
    }

    const uint64_t arlenb = alenb / rows;
    const uint64_t elems = rows * rows;
    const uint64_t alen_bits = alenb * 8;
    if (elems == 0 || alen_bits % elems != 0) {
        printf("[xsai-ame] unavailable: tlenb=%llu trlenb=%llu alenb=%llu rows=%llu\n",
            (unsigned long long) tlenb,
            (unsigned long long) trlenb,
            (unsigned long long) alenb,
            (unsigned long long) rows);
        return 0;
    }

    const uint64_t melen_bits = alen_bits / elems;
    const uint64_t int8_m_max = rows;
    const uint64_t int8_k_max = trlenb;
    const uint64_t int32_n_max = arlenb / sizeof(int32_t);
    const char * tag = int8_capacity_tag(int8_m_max, int8_k_max, int32_n_max);

    printf("[xsai-ame] tlenb=%llu trlenb=%llu alenb=%llu rows=%llu melen_bits=%llu arlenb=%llu int8_capacity=%s\n",
        (unsigned long long) tlenb,
        (unsigned long long) trlenb,
        (unsigned long long) alenb,
        (unsigned long long) rows,
        (unsigned long long) melen_bits,
        (unsigned long long) arlenb,
        tag);

    return 0;
}
