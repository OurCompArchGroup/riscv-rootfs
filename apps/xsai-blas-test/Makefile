# xsai-blas test cases as a Linux program for the rootfs. xsai-blas builds its
# libraries with CMake; this app only puts a main() in front of them.
#
#   make              # build/xsai-blas-test
#   make install      # copy it into the rootfs image
#
#   xsai-blas-test [test|perf] [--kernel NAME] [--case ID]

NAME = xsai-blas-test
SRCS = main.c

XS_PROJECT_ROOT ?= $(abspath $(RISCV_ROOTFS_HOME)/../..)
XSAI_BLAS_HOME  ?= $(XS_PROJECT_ROOT)/xsai-blas
# i8 GEMM tile M/N (64 or 128); K stays 64. One library build per tile.
TILE_M          ?= 128
TILE_N          ?= 128
XSAI_BLAS_BUILD ?= $(XSAI_BLAS_HOME)/build/linux-test-tile$(TILE_M)x$(TILE_N)
XSAI_BLAS_LIBS  := $(XSAI_BLAS_BUILD)/xsai-blas-tests.a \
                   $(XSAI_BLAS_BUILD)/xsai-blas-riscv64-xs.a

INC_DIR += $(XSAI_BLAS_HOME)/test/include

include $(RISCV_ROOTFS_HOME)/Makefile.compile

# Same compiler as the xsai-blas archives; sysroot and libgcc come from the
# environment's RISC-V Linux toolchain. A Debian multiarch directory such as
# /usr/riscv64-linux-gnu is found through --gcc-toolchain and must not be
# passed as --sysroot (its linker scripts use absolute paths).
RISCV_TARGET_FLAGS := --target=riscv64-unknown-linux-gnu -march=rv64gc \
                      --gcc-toolchain=$(RISCV) \
                      $(if $(wildcard $(RISCV_SYSROOT)/usr),--sysroot=$(RISCV_SYSROOT))
override CC := $(LLVM_HOME)/bin/clang
override LD := $(LLVM_HOME)/bin/clang
CFLAGS  += $(RISCV_TARGET_FLAGS)
LDFLAGS += $(RISCV_TARGET_FLAGS)

# Makefile.compile links $(OBJS) after $(LDFLAGS), so the archives go after the
# objects that use them.
OBJS += $(XSAI_BLAS_LIBS) -lm -lpthread

xsai-blas:
	@cd $(XSAI_BLAS_HOME) && cmake --preset linux-xsai -B $(XSAI_BLAS_BUILD) \
	  -DXSAI_BLAS_BUILD_TEST_CASES=ON \
	  -DXSAI_BLAS_TILE_M=$(TILE_M) -DXSAI_BLAS_TILE_N=$(TILE_N) \
	  > /dev/null
	@cmake --build $(XSAI_BLAS_BUILD) -j

$(APP): xsai-blas

.PHONY: xsai-blas
