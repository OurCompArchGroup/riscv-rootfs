// nemu-signal: emit the checkpoint-flow custom instruction (0x6B) with a0 set
// to the requested nemu_signal value. See docs/checkpoint-workload-template.md
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: nemu-signal <value>\n");
        return 2;
    }
    long value = strtol(argv[1], NULL, 0);
    register long a0 asm("a0") = value;
    __asm__ volatile(".insn r 0x6B, 0, 0, x0, x0, x0" : : "r"(a0) : "memory");
    return 0;
}
