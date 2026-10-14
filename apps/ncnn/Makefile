# XSAI rootfs app: Tencent/ncnn — AME int8 conv port scaffold (phase 1).
#
# Mirrors apps/mnn:
#   source     — $(XSAI_ENV_ROOT)/software/ncnn (explicit NCNN_SRC override only)
#   build-riscv/libame_xsa.a — the single AME TU (ame_xsa.c from the MNN tree,
#                              NOT a copy) + xsai pool allocator, zames march
#   build-riscv/ncnn-build/  — cross-compiled ncnn static lib + test binaries
#   build-riscv/ncnn_ame_smoke — differential smoke (AME=1 vs AME=0)
#
# The AME kernel enters ncnn through src/layer/riscv/convolution_int8_xsai.h
# (see software/ncnn), compiled with -DNCNN_XSAI_AME=1 and linked against
# libame_xsa.a via CMAKE_CXX_STANDARD_LIBRARIES.
#
# NOTE: NCNN_ZFH=OFF — the stock fp16sa RVV gemm hits an illegal instruction
# under qemu-user (x-matrix fork, vfh); fp16 kernels are out of phase-1 scope.

NAME := ncnn

THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
APP_DIR := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
XSAI_ENV_ROOT ?= $(abspath $(APP_DIR)/../../../..)
XSAI_RESOLVE_RISCV_ENV := $(XSAI_ENV_ROOT)/scripts/resolve-riscv-env.sh
RISCV_ROOTFS_HOME ?= $(XSAI_ENV_ROOT)/firmware/riscv-rootfs
ROOTFS_BUILD_DIR ?= $(RISCV_ROOTFS_HOME)/rootfsimg/build

RISCV ?= /opt/riscv
LLVM_HOME ?= $(XSAI_ENV_ROOT)/llvm-project-ame/build
export PATH := $(RISCV)/bin:$(PATH)

CMAKE ?= cmake
NPROC ?= $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)

# Source acquisition/version locks belong to the parent workspace.
# *_HOME remain explicit legacy aliases; *_SRC take precedence.
NCNN_DEFAULT_HOME := $(XSAI_ENV_ROOT)/software/ncnn
NCNN_SRC ?= $(or $(NCNN_HOME),$(NCNN_DEFAULT_HOME))
override NCNN_SRC := $(abspath $(NCNN_SRC))
override NCNN_HOME := $(NCNN_SRC)
# Explicit dependency: ncnn links the AME kernel AND allocator from MNN.
MNN_SRC ?= $(or $(MNN_HOME),$(XSAI_ENV_ROOT)/software/MNN)
override MNN_SRC := $(abspath $(MNN_SRC))
override MNN_HOME := $(MNN_SRC)
MNN_AME_DIR := $(MNN_HOME)/source/backend/cpu/ame
MNN_COMPUTE_DIR := $(MNN_HOME)/source/backend/cpu/compute

TARGET_BUILD_DIR ?= $(APP_DIR)/build-riscv
NCNN_BUILD_DIR := $(TARGET_BUILD_DIR)/ncnn-build
NCNN_LIB := $(NCNN_BUILD_DIR)/src/libncnn.a
AME_LIB := $(TARGET_BUILD_DIR)/libame_xsa.a
SMOKE_BIN := $(TARGET_BUILD_DIR)/ncnn_ame_smoke
NCNN_TEST_CONV := $(NCNN_BUILD_DIR)/tests/test_convolution_3
ROOTFS_SMOKE_BIN := $(ROOTFS_BUILD_DIR)/ncnn_ame_smoke

NCNN_TOOLCHAIN_FILE := $(APP_DIR)/cmake/riscv64-toolchain.cmake
RISCV_TRIPLE ?= riscv64-unknown-linux-gnu
AME_MARCH ?= rv64gcv_zba_zicbop_zames_zmasync_zvl128b
NCNN_MARCH ?= rv64gcv_zba_zicbop
RISCV_MABI ?= lp64d
NCNN_RISCV_SYSROOT ?= $(strip $(or $(RISCV_SYSROOT),$(QEMU_LD_PREFIX),$(wildcard $(RISCV)/sysroot),$(shell [ -x "$(XSAI_RESOLVE_RISCV_ENV)" ] && "$(XSAI_RESOLVE_RISCV_ENV)" sysroot 2>/dev/null || true),$(shell $(RISCV_TRIPLE)-gcc -print-sysroot 2>/dev/null),$(RISCV)/sysroot))
ifeq ($(strip $(RISCV_SYSROOT)),)
RISCV_SYSROOT := $(NCNN_RISCV_SYSROOT)
endif

# AME operand pool 的编译期兜底（默认 appliance 8GB 布局）。
# 运行时优先 XSAI_AME_RESERVED_BASE 环境变量 → DTB，mega(16GB/4GB, 池@0x380000000) 用：
#   make ... XSAI_ALLOC_BASE=0x380000000ULL XSAI_ALLOC_SIZE='\(4096L\*1024L\*1024L\)'
XSAI_ALLOC_BASE ?= 0x100000000ULL
XSAI_ALLOC_SIZE ?= \(64L\*1024L\*1024L\)
XSAI_ALLOC_DEFS := -DRESERVED_PHYS_BASE_ADDR=$(XSAI_ALLOC_BASE) -DRESERVED_MEMORY_SIZE=$(XSAI_ALLOC_SIZE)

QEMU_HOME ?= $(XSAI_ENV_ROOT)/qemu
QEMU_USER_CPU_FLAGS ?= rv64,v=true,vlen=128,h=false,zvfh=true,zvfhmin=true,zvfbfmin=true,zvfbfwma=true,x-matrix=true,rlen=512,mlen=65536,melen=32
QEMU_USER_PREFIX := $(strip $(or $(QEMU_LD_PREFIX),$(RISCV_SYSROOT)))
VLOG_DIR ?= $(TARGET_BUILD_DIR)/logs

.DEFAULT_GOAL := help

.PHONY: help repo source-ready mnn-source-ready ame-lib riscv smoke test-binaries all run-user-smoke run-user-test install run-model run-user-e2e print-config clean

help:
	@echo "ncnn xsai wrapper (phase 1: int8 conv AME port)"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn source-ready   validate ncnn and required MNN AME sources"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn repo           compatibility alias for source-ready; no downloads"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn ame-lib        build libame_xsa.a (zames march TU)"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn riscv          cross-build ncnn static lib"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn test-binaries  build official test_convolution_3"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn smoke          build differential smoke binary"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn run-user-smoke  qemu-user AME-vs-ref diff"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn run-user-test   qemu-user official test_convolution_3"
	@echo "  make -C firmware/riscv-rootfs/apps/ncnn install        copy smoke binary into rootfs build dir"
	@echo "env: RISCV=$(RISCV) LLVM_HOME=$(LLVM_HOME)"
	@echo "     NCNN_SRC=$(NCNN_SRC) MNN_SRC=$(MNN_SRC) (*_HOME are legacy override aliases)"

repo: source-ready

source-ready: mnn-source-ready
	@if [ ! -f "$(NCNN_HOME)/CMakeLists.txt" ]; then \
		echo "error: ncnn source missing at $(NCNN_SRC)" >&2; \
		echo "hint: acquire the locked source at $(NCNN_DEFAULT_HOME) via the parent workspace, or set NCNN_SRC=/path/to/ncnn explicitly" >&2; \
		exit 1; \
	fi

mnn-source-ready:
	@for src in "$(MNN_COMPUTE_DIR)/ame_xsa.c" "$(MNN_COMPUTE_DIR)/ame_xsa.h" "$(MNN_AME_DIR)/xsai_alloc.c" "$(MNN_AME_DIR)/xsai_alloc.h"; do \
		test -f "$$src" || { \
			echo "error: ncnn requires the MNN AME kernel/allocator: $$src" >&2; \
			echo "hint: acquire locked software/MNN via the parent workspace, or set MNN_SRC=/path/to/MNN explicitly" >&2; \
			exit 1; \
		}; \
	done

# The only TU with AME instructions: ame_xsa.c from the MNN tree, compiled
# with the AME-aware assembler in LLVM_HOME (zames/zmasync/zvl128b march).
$(AME_LIB): mnn-source-ready $(wildcard $(MNN_COMPUTE_DIR)/ame_xsa.c $(MNN_COMPUTE_DIR)/ame_xsa.h $(MNN_AME_DIR)/xsai_alloc.c $(MNN_AME_DIR)/xsai_alloc.h)
	@for tool in clang llvm-ar ld.lld; do \
		test -x "$(LLVM_HOME)/bin/$$tool" || { echo "error: AME LLVM tool missing: $(LLVM_HOME)/bin/$$tool; set LLVM_HOME" >&2; exit 1; }; \
	done
	mkdir -p $(TARGET_BUILD_DIR)
	$(LLVM_HOME)/bin/clang --target=$(RISCV_TRIPLE) --sysroot=$(RISCV_SYSROOT) \
		--gcc-toolchain=$(RISCV) -march=$(AME_MARCH) -mabi=$(RISCV_MABI) -O2 \
		$(XSAI_ALLOC_DEFS) -I$(MNN_AME_DIR) -I$(MNN_COMPUTE_DIR) \
		-c $(MNN_COMPUTE_DIR)/ame_xsa.c -o $(TARGET_BUILD_DIR)/ame_xsa.o
	$(LLVM_HOME)/bin/clang --target=$(RISCV_TRIPLE) --sysroot=$(RISCV_SYSROOT) \
		--gcc-toolchain=$(RISCV) -march=$(AME_MARCH) -mabi=$(RISCV_MABI) -O2 \
		$(XSAI_ALLOC_DEFS) -I$(MNN_AME_DIR) -I$(MNN_COMPUTE_DIR) \
		-c $(MNN_AME_DIR)/xsai_alloc.c -o $(TARGET_BUILD_DIR)/xsai_alloc.o
	$(LLVM_HOME)/bin/llvm-ar rcs $@ $(TARGET_BUILD_DIR)/ame_xsa.o $(TARGET_BUILD_DIR)/xsai_alloc.o

ame-lib: $(AME_LIB)

# ncnn cross build. libame_xsa.a rides along on every link via
# CMAKE_CXX_STANDARD_LIBRARIES (appended after libncnn.a), and the AME glue
# in src/layer/riscv/ is activated with -DNCNN_XSAI_AME=1 + the ame_xsa.h
# include paths from the MNN tree (single-implementation discipline).
$(NCNN_LIB): source-ready $(AME_LIB) $(wildcard $(NCNN_HOME)/src/*.cpp $(NCNN_HOME)/src/layer/riscv/*)
	mkdir -p $(NCNN_BUILD_DIR)
	cd $(NCNN_BUILD_DIR) && $(CMAKE) $(NCNN_HOME) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_TOOLCHAIN_FILE=$(NCNN_TOOLCHAIN_FILE) \
		-DRISCV_TRIPLE=$(RISCV_TRIPLE) \
		-DRISCV_ROOT_PATH=$(LLVM_HOME) \
		-DRISCV_MARCH=$(NCNN_MARCH) \
		-DRISCV_MABI=$(RISCV_MABI) \
		-DRISCV_USE_LLVM=ON \
		-DRISCV_SYSROOT=$(RISCV_SYSROOT) \
		-DCMAKE_CXX_FLAGS="-DNCNN_XSAI_AME=1 -I$(MNN_COMPUTE_DIR) -I$(MNN_AME_DIR)" \
		-DCMAKE_CXX_STANDARD_LIBRARIES=$(AME_LIB) \
		-DNCNN_BUILD_TOOLS=OFF \
		-DNCNN_BUILD_EXAMPLES=OFF \
		-DNCNN_BUILD_TESTS=ON \
		-DNCNN_RUNTIME_CPU=OFF \
		-DNCNN_OPENMP=OFF \
		-DNCNN_ZFH=OFF \
		-DNCNN_INT8=ON
	$(CMAKE) --build $(NCNN_BUILD_DIR) -j $(NPROC) --target ncnn

riscv: $(NCNN_LIB)

# official per-op test binaries (int8 conv cases live in test_convolution_3)
test-binaries: $(NCNN_TEST_CONV)

$(NCNN_TEST_CONV): $(NCNN_LIB)
	$(CMAKE) --build $(NCNN_BUILD_DIR) -j $(NPROC) --target test_convolution_3

RUN_MODEL_BIN := $(TARGET_BUILD_DIR)/ncnn_run_model
MODELS_DIR ?= $(APP_DIR)/models
ROOTFS_RUN_MODEL := $(ROOTFS_BUILD_DIR)/ncnn_run_model

$(RUN_MODEL_BIN): $(APP_DIR)/ncnn_run_model.cpp $(NCNN_LIB) $(AME_LIB)
	$(LLVM_HOME)/bin/clang++ --target=$(RISCV_TRIPLE) --sysroot=$(RISCV_SYSROOT) \
		--gcc-toolchain=$(RISCV) -march=$(NCNN_MARCH) -mabi=$(RISCV_MABI) -O2 -static \
		-DNCNN_XSAI_AME=1 \
		-I$(NCNN_HOME)/src -I$(NCNN_HOME)/src/layer -I$(NCNN_BUILD_DIR)/src \
		-I$(MNN_COMPUTE_DIR) -I$(MNN_AME_DIR) \
		-fuse-ld=lld --ld-path=$(LLVM_HOME)/bin/ld.lld -o $@ \
		$(APP_DIR)/ncnn_run_model.cpp $(NCNN_LIB) $(AME_LIB) -lpthread -lm

run-model: $(RUN_MODEL_BIN)

# qemu E2E: fp32 model with the bf16 AME path on/off + squeezenet sanity.
# Ground truth = onnxruntime on the host (mobilenet argmax 282, squeezenet 669).
run-user-e2e: $(RUN_MODEL_BIN)
	@mkdir -p $(TARGET_BUILD_DIR)/run
	cd $(TARGET_BUILD_DIR)/run && cp -f $(RUN_MODEL_BIN) . && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon XSAI_NCNN_AME_BF16=0 \
	$(QEMU_HOME)/build/qemu-riscv64 -cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_run_model \
		$(MODELS_DIR)/mobilenet-v2.param $(MODELS_DIR)/mobilenet-v2.bin $(ROOTFS_BUILD_DIR)/mnn-input-0.txt 224 224 3
	cd $(TARGET_BUILD_DIR)/run && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon XSAI_NCNN_AME_BF16=1 \
	$(QEMU_HOME)/build/qemu-riscv64 -cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_run_model \
		$(MODELS_DIR)/mobilenet-v2.param $(MODELS_DIR)/mobilenet-v2.bin $(ROOTFS_BUILD_DIR)/mnn-input-0.txt 224 224 3
	cd $(TARGET_BUILD_DIR)/run && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon XSAI_NCNN_AME_BF16=0 \
	$(QEMU_HOME)/build/qemu-riscv64 -cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_run_model \
		$(MODELS_DIR)/squeezenet.param $(MODELS_DIR)/squeezenet.bin $(ROOTFS_BUILD_DIR)/mnn-input-0.txt 224 224 3
	cd $(TARGET_BUILD_DIR)/run && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon XSAI_NCNN_AME_BF16=1 \
	$(QEMU_HOME)/build/qemu-riscv64 -cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_run_model \
		$(MODELS_DIR)/squeezenet.param $(MODELS_DIR)/squeezenet.bin $(ROOTFS_BUILD_DIR)/mnn-input-0.txt 224 224 3

$(SMOKE_BIN): $(APP_DIR)/ncnn_ame_smoke.cpp $(NCNN_LIB) $(AME_LIB)
	$(LLVM_HOME)/bin/clang++ --target=$(RISCV_TRIPLE) --sysroot=$(RISCV_SYSROOT) \
		--gcc-toolchain=$(RISCV) -march=$(NCNN_MARCH) -mabi=$(RISCV_MABI) -O2 -static \
		-DNCNN_XSAI_AME=1 \
		-I$(NCNN_HOME)/src -I$(NCNN_HOME)/src/layer -I$(NCNN_BUILD_DIR)/src \
		-I$(MNN_COMPUTE_DIR) -I$(MNN_AME_DIR) \
		-fuse-ld=lld --ld-path=$(LLVM_HOME)/bin/ld.lld -o $@ \
		$(APP_DIR)/ncnn_ame_smoke.cpp $(NCNN_LIB) $(AME_LIB) -lpthread -lm

smoke: $(SMOKE_BIN)

all: riscv test-binaries smoke

# differential smoke: same binary, env flips the AME gates.
#   int8: outputs must be bit-identical (same epilogue arithmetic)
#   bf16: tight few-ulp gate in-process vs scalar bf16 ref (exit code), plus
#         this loose cross-process compare vs the stock fp32 conv
run-user-smoke: $(SMOKE_BIN)
	@mkdir -p $(TARGET_BUILD_DIR)/run $(VLOG_DIR)
	@rm -f $(TARGET_BUILD_DIR)/run/*.out
	cd $(TARGET_BUILD_DIR)/run && cp -f $(SMOKE_BIN) . && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon $(QEMU_HOME)/build/qemu-riscv64 \
		-cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_ame_smoke ref ref.out ref-bf16.out
	cd $(TARGET_BUILD_DIR)/run && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_ALLOC_MODE=anon $(QEMU_HOME)/build/qemu-riscv64 \
		-cpu $(QEMU_USER_CPU_FLAGS) ./ncnn_ame_smoke ame ame.out ame-bf16.out
	@if test -s $(TARGET_BUILD_DIR)/run/ref.out && test -s $(TARGET_BUILD_DIR)/run/ame.out && cmp -s $(TARGET_BUILD_DIR)/run/ref.out $(TARGET_BUILD_DIR)/run/ame.out; then \
		echo "[ncnn-smoke] int8 PASS: AME output == reference ($$(wc -l < $(TARGET_BUILD_DIR)/run/ame.out) lines)"; \
	else \
		echo "[ncnn-smoke] int8 FAIL: outputs differ"; \
		cmp $(TARGET_BUILD_DIR)/run/ref.out $(TARGET_BUILD_DIR)/run/ame.out | head -10; exit 1; \
	fi
	python3 $(APP_DIR)/compare_smoke_bf16.py $(TARGET_BUILD_DIR)/run/ref-bf16.out $(TARGET_BUILD_DIR)/run/ame-bf16.out

# official gate: the whole test_convolution_3 suite under qemu, AME forced on.
run-user-test: $(NCNN_TEST_CONV)
	@mkdir -p $(TARGET_BUILD_DIR)/run
	cd $(TARGET_BUILD_DIR)/run && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) XSAI_NCNN_AME=1 XSAI_ALLOC_MODE=anon \
	$(QEMU_HOME)/build/qemu-riscv64 -cpu $(QEMU_USER_CPU_FLAGS) $(NCNN_TEST_CONV)

ROOTFS_TEST_CONV := $(ROOTFS_BUILD_DIR)/ncnn_test_convolution_3

# Board payload inputs: the static smoke binary (self-contained differential
# gate) plus the official per-op test binary (dynamic, same NEEDED set as
# MNNV2Basic.out which already runs on board). Both land in rootfsimg/build
# and the initramfs manifest ships them as /bin/*.
# Model-free differential smoke and runners. External models are opt-in only.
install: $(SMOKE_BIN) $(RUN_MODEL_BIN) $(NCNN_TEST_CONV)
	mkdir -p $(ROOTFS_BUILD_DIR)
	cp -f $(SMOKE_BIN) $(ROOTFS_SMOKE_BIN)
	cp -f $(RUN_MODEL_BIN) $(ROOTFS_RUN_MODEL)
	cp -f $(NCNN_TEST_CONV) $(ROOTFS_TEST_CONV)
	@ls -la $(ROOTFS_SMOKE_BIN) $(ROOTFS_RUN_MODEL) $(ROOTFS_TEST_CONV)

print-config:
	@echo "NCNN_SRC      = $(NCNN_SRC)"
	@echo "MNN_SRC       = $(MNN_SRC)"
	@echo "NCNN_HOME     = $(NCNN_HOME)"
	@echo "MNN_HOME      = $(MNN_HOME)"
	@echo "LLVM_HOME     = $(LLVM_HOME)"
	@echo "RISCV         = $(RISCV)"
	@echo "RISCV_SYSROOT = $(RISCV_SYSROOT)"
	@echo "AME lib       = $(AME_LIB)"
	@echo "ncnn lib      = $(NCNN_LIB)"

clean:
	rm -rf $(TARGET_BUILD_DIR)
