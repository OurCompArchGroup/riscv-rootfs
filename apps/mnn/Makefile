# XSAI rootfs app: alibaba/MNN — RVV inference smoke (MobileNetV2 int8).
#
# Mirrors the apps/llama.cpp structure:
#   source     — $(XSAI_ENV_ROOT)/software/MNN (explicit MNN_SRC override only)
#   build-host — host tools: MNNConvert, quantized.out (int8 PTQ)
#   build-riscv— target build: MNN static + MNNV2Basic.out
#   models     — ONNX source, fp32/int8 .mnn, synthetic calibration BMPs
#
# Model pipeline: ONNX -> fp32 .mnn (MNNConvert) -> int8 .mnn (quantized.out
# with deterministic synthetic calibration images, see scripts/gen-calib.py).

NAME := mnn

THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
APP_DIR := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
XSAI_ENV_ROOT ?= $(abspath $(APP_DIR)/../../../..)
XSAI_RESOLVE_RISCV_ENV := $(XSAI_ENV_ROOT)/scripts/resolve-riscv-env.sh
RISCV_ROOTFS_HOME ?= $(XSAI_ENV_ROOT)/firmware/riscv-rootfs
ROOTFS_BUILD_DIR ?= $(RISCV_ROOTFS_HOME)/rootfsimg/build
# Memory layout include kept for parity with the other apps; unused directly.
include $(XSAI_ENV_ROOT)/mk/memory.mk

RISCV ?= /opt/riscv
LLVM_HOME ?= $(XSAI_ENV_ROOT)/llvm-project-ame/build
export PATH := $(RISCV)/bin:$(PATH)

CMAKE ?= cmake
NPROC ?= $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)

# Source acquisition/version locks belong to the parent workspace.
# MNN_HOME remains an explicit legacy alias; MNN_SRC takes precedence.
MNN_DEFAULT_HOME := $(XSAI_ENV_ROOT)/software/MNN
MNN_SRC ?= $(or $(MNN_HOME),$(MNN_DEFAULT_HOME))
override MNN_SRC := $(abspath $(MNN_SRC))
override MNN_HOME := $(MNN_SRC)

# Host tools (x86_64): model conversion + PTQ.
HOST_CC ?= cc
HOST_CXX ?= c++
HOST_NO_CET_FLAG ?= -fcf-protection=none
HOST_BUILD_DIR ?= $(APP_DIR)/build-host
HOST_CONVERT_BIN := $(HOST_BUILD_DIR)/MNNConvert
HOST_QUANT_BIN := $(HOST_BUILD_DIR)/quantized.out
HOST_TOOLS_STAMP := $(HOST_BUILD_DIR)/.tools-built

# Target build (riscv64): MNN static + inference runner.
TARGET_BUILD_DIR ?= $(APP_DIR)/build-riscv
TARGET_V2BASIC_BIN := $(TARGET_BUILD_DIR)/MNNV2Basic.out
TARGET_GETINFO_BIN := $(TARGET_BUILD_DIR)/GetMNNInfo
TARGET_SIGNAL_BIN := $(TARGET_BUILD_DIR)/nemu-signal
TARGET_AMETEST_BIN := $(TARGET_BUILD_DIR)/test_ame_kernels
TARGET_STAMP := $(TARGET_BUILD_DIR)/.mnn-built
ROOTFS_V2BASIC_BIN := $(ROOTFS_BUILD_DIR)/MNNV2Basic.out
ROOTFS_GETINFO_BIN := $(ROOTFS_BUILD_DIR)/GetMNNInfo
ROOTFS_SIGNAL_BIN := $(ROOTFS_BUILD_DIR)/nemu-signal
ROOTFS_AMETEST_BIN := $(ROOTFS_BUILD_DIR)/test_ame_kernels

MNN_TOOLCHAIN_FILE := $(APP_DIR)/cmake/riscv64-toolchain.cmake
RISCV_TRIPLE ?= riscv64-unknown-linux-gnu
AME_MARCH ?= rv64gcv_zba_zicbop_zames_zmasync_zvl128b
RISCV_MARCH ?= rv64gc_zba_zicbop
RISCV_MABI ?= lp64d
# Same sysroot resolution order as apps/llama.cpp.
MNN_RISCV_SYSROOT ?= $(strip $(or $(RISCV_SYSROOT),$(QEMU_LD_PREFIX),$(shell [ -x "$(XSAI_RESOLVE_RISCV_ENV)" ] && "$(XSAI_RESOLVE_RISCV_ENV)" sysroot 2>/dev/null || true),$(shell $(RISCV_TRIPLE)-gcc -print-sysroot 2>/dev/null),$(RISCV)/sysroot,/opt/riscv/sysroot))
ifeq ($(strip $(RISCV_SYSROOT)),)
RISCV_SYSROOT := $(MNN_RISCV_SYSROOT)
endif

# nemu-signal: raw 0x6B custom instruction emitter for checkpoint ROI markers.
CROSS_GCC := $(shell command -v $(RISCV_TRIPLE)-gcc 2>/dev/null || command -v riscv64-linux-gnu-gcc 2>/dev/null)

$(TARGET_SIGNAL_BIN): $(APP_DIR)/nemu-signal.c
	@command -v "$(CROSS_GCC)" >/dev/null 2>&1 || { echo "error: RISC-V GCC not found; set RISCV or CROSS_GCC" >&2; exit 1; }
	mkdir -p $(TARGET_BUILD_DIR)
	$(CROSS_GCC) -O2 -static -o $@ $<

# AME GEMM correctness/cycle harness (int8 + bf16). Compiles the real kernel
# from the MNN tree, so the harness and MNN share one implementation. Needs the
# AME-aware assembler in LLVM_HOME (llvm-project-ame build) for the
# msettilem/mla/mmacc/msc family; the AME instruction set requires the
# _zames_zmasync_zvl128b march suffix (v included) and the gcc15 toolchain for
# crt/libgcc.
XSAI_ALLOC_DEFS := -DRESERVED_PHYS_BASE_ADDR=0x100000000ULL -DRESERVED_MEMORY_SIZE=\(64L\*1024L\*1024L\)
AME_KERNEL_SRC := $(MNN_HOME)/source/backend/cpu/compute/ame_xsa.c

$(TARGET_AMETEST_BIN): source-ready $(APP_DIR)/test_ame_kernels.c $(wildcard $(AME_KERNEL_SRC) $(AME_KERNEL_SRC:.c=.h) $(MNN_HOME)/source/backend/cpu/ame/xsai_alloc.c $(MNN_HOME)/source/backend/cpu/ame/xsai_alloc.h)
	@test -x "$(LLVM_HOME)/bin/clang" -a -x "$(LLVM_HOME)/bin/ld.lld" || { echo "error: AME LLVM clang/ld.lld missing; set LLVM_HOME" >&2; exit 1; }
	mkdir -p $(TARGET_BUILD_DIR)
	$(LLVM_HOME)/bin/clang --target=$(RISCV_TRIPLE) --sysroot=$(MNN_RISCV_SYSROOT) \
		--gcc-toolchain=$(RISCV) -march=$(AME_MARCH) -mabi=$(RISCV_MABI) -O2 -static \
		$(XSAI_ALLOC_DEFS) -I$(MNN_HOME)/source/backend/cpu/ame -I$(MNN_HOME)/source/backend/cpu/compute \
		-fuse-ld=lld --ld-path=$(LLVM_HOME)/bin/ld.lld -o $@ \
		$(APP_DIR)/test_ame_kernels.c $(AME_KERNEL_SRC) $(MNN_HOME)/source/backend/cpu/ame/xsai_alloc.c

MNN_COMMON_CMAKE_ARGS = \
	-DCMAKE_BUILD_TYPE=Release \
	-DMNN_BUILD_SHARED_LIBS=OFF \
	-DMNN_BUILD_DEMO=OFF \
	-DMNN_BUILD_TRAIN=OFF \
	-DMNN_BUILD_BENCHMARK=OFF \
	-DMNN_BUILD_TEST=OFF \
	-DMNN_BUILD_LLM=OFF \
	-DMNN_BUILD_DIFFUSION=OFF \
	-DMNN_BUILD_OPENCV=OFF \
	-DMNN_BUILD_AUDIO=OFF \
	-DMNN_SUPPORT_BF16=ON \
	-DMNN_SUPPORT_RENDER=OFF \
	-DMNN_OPENMP=OFF \
	-DMNN_USE_THREAD_POOL=ON \
	-DMNN_KLEIDIAI=OFF

HOST_CMAKE_ARGS = $(MNN_COMMON_CMAKE_ARGS) \
	-DCMAKE_C_COMPILER=$(HOST_CC) \
	-DCMAKE_CXX_COMPILER=$(HOST_CXX) \
	-DCMAKE_C_FLAGS="$(HOST_NO_CET_FLAG)" \
	-DCMAKE_CXX_FLAGS="$(HOST_NO_CET_FLAG)" \
	-DMNN_BUILD_CONVERTER=ON \
	-DMNN_BUILD_QUANTOOLS=ON \
	-DMNN_BUILD_TOOLS=OFF

TARGET_CMAKE_ARGS = $(MNN_COMMON_CMAKE_ARGS) \
	-DCMAKE_TOOLCHAIN_FILE=$(MNN_TOOLCHAIN_FILE) \
	-DRISCV_TRIPLE=$(RISCV_TRIPLE) \
	-DRISCV_ROOT_PATH=$(LLVM_HOME) \
	-DRISCV_MARCH=$(RISCV_MARCH) \
	-DRISCV_MABI=$(RISCV_MABI) \
	-DRISCV_USE_LLVM=ON \
	-DRISCV_SYSROOT=$(MNN_RISCV_SYSROOT) \
	-DMNN_BUILD_CONVERTER=OFF \
	-DMNN_BUILD_QUANTOOLS=OFF \
	-DMNN_BUILD_TOOLS=ON \
	-DMNN_USE_RVV=ON

# Model pipeline.
MODELS_DIR ?= $(APP_DIR)/models
# MNNV2Basic reads pwd/input_0.txt (text floats) and writes pwd/output.txt
# next to the binary. Each smoke variant runs from its own scratch dir under
# RUN_SMOKE_DIR so fp32/int8 outputs never collide. Definitions live above
# the install rule: prerequisite lists expand at parse time.
RUN_SMOKE_DIR := $(APP_DIR)/build-smoke
INPUT_FILE := $(RUN_SMOKE_DIR)/cat-input.txt
# Model acquisition is explicit and hash-checked. No mutable main/latest URL.
# An existing ONNX_FILE can be supplied without any network access.
MOBILENET_V2_ONNX_URL ?=
MOBILENET_V2_ONNX_SHA256 ?=
ONNX_FILE ?= $(MODELS_DIR)/mobilenetv2-12.onnx
FP32_MODEL := $(MODELS_DIR)/mobilenet-v2-fp32.mnn
INT8_MODEL := $(MODELS_DIR)/mobilenet-v2-int8.mnn
CALIB_DIR := $(MODELS_DIR)/calib
CALIB_IMAGES := $(CALIB_DIR)/xsai-calib-000.bmp
QUANT_CONFIG := $(HOST_BUILD_DIR)/mobilenet-v2-quant.json
ROOTFS_INT8_MODEL := $(ROOTFS_BUILD_DIR)/mobilenet-v2-int8.mnn
ROOTFS_FP32_MODEL := $(ROOTFS_BUILD_DIR)/mobilenet-v2-fp32.mnn
ROOTFS_FP16_MODEL := $(ROOTFS_BUILD_DIR)/mobilenet-v2-fp16.mnn
ROOTFS_SQ_MODEL := $(ROOTFS_BUILD_DIR)/squeezenet-int8.mnn
FP16_MODEL := $(MODELS_DIR)/mobilenet-v2-fp16.mnn
SQ_MODEL := $(MODELS_DIR)/squeezenet-int8.mnn
ROOTFS_INPUT := $(ROOTFS_BUILD_DIR)/mnn-input-0.txt

QEMU_HOME ?= $(XSAI_ENV_ROOT)/qemu
QEMU_USER_CPU_FLAGS ?= rv64,v=true,vlen=128,h=false,zvfh=true,zvfhmin=true,zvfbfmin=true,zvfbfwma=true,x-matrix=true,rlen=512,mlen=65536,melen=32
QEMU_USER_PREFIX := $(strip $(or $(QEMU_LD_PREFIX),$(RISCV_SYSROOT)))

# MNNV2Basic.out args: model runLoops runMask forwardType numberThread precision inputSize
V2BASIC_ARGS ?= $(ROOTFS_INT8_MODEL) 10 0 0 1 0 1x3x224x224

.DEFAULT_GOAL := install

.PHONY: repo source-ready model-source-ready host-tools model fp32-model int8-model riscv install install-binaries \
	rootfs run-user run-user-fp32 ame-harness print-config clean distclean help

help:
	@echo "mnn xsai wrapper"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn source-ready validate selected MNN source"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn repo         compatibility alias for source-ready; no downloads"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn host-tools   build MNNConvert + quantized.out"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn model        onnx -> fp32.mnn -> int8.mnn"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn riscv        cross-build MNNV2Basic.out"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn ame-harness  AME int8+bf16 kernel harness"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn install      install binary + model into rootfs"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn install-binaries  model-free build/install gate"
	@echo "  make -C firmware/riscv-rootfs/apps/mnn run-user     qemu-riscv64 smoke test"
	@echo "source: MNN_SRC=$(MNN_SRC) (MNN_HOME is a legacy override alias)"
	@echo "env: needs RISCV/RISCV_SYSROOT/LLVM_HOME (see .envrc.local); model downloads need network"

repo: source-ready

ame-harness: source-ready $(TARGET_AMETEST_BIN)

source-ready:
	@if [ ! -f "$(MNN_HOME)/CMakeLists.txt" ]; then \
		echo "error: MNN source missing at $(MNN_SRC)" >&2; \
		echo "hint: acquire the locked source at $(MNN_DEFAULT_HOME) via the parent workspace, or set MNN_SRC=/path/to/MNN explicitly" >&2; \
		exit 1; \
	fi
	@for src in source/backend/cpu/compute/ame_xsa.c source/backend/cpu/compute/ame_xsa.h source/backend/cpu/ame/xsai_alloc.c source/backend/cpu/ame/xsai_alloc.h; do \
		test -f "$(MNN_SRC)/$$src" || { echo "error: required XSAI MNN source missing: $(MNN_SRC)/$$src" >&2; exit 1; }; \
	done

$(HOST_TOOLS_STAMP): source-ready
	mkdir -p $(HOST_BUILD_DIR)
	cd $(HOST_BUILD_DIR) && $(CMAKE) $(MNN_HOME) $(HOST_CMAKE_ARGS)
	$(CMAKE) --build $(HOST_BUILD_DIR) -j $(NPROC) --target MNNConvert quantized.out
	@touch $@

host-tools: $(HOST_TOOLS_STAMP)

$(HOST_CONVERT_BIN) $(HOST_QUANT_BIN): $(HOST_TOOLS_STAMP)
	@test -x "$@"

$(TARGET_STAMP): source-ready
	@test -x "$(LLVM_HOME)/bin/clang" || { echo "error: AME LLVM clang missing; set LLVM_HOME" >&2; exit 1; }
	mkdir -p $(TARGET_BUILD_DIR)
	cd $(TARGET_BUILD_DIR) && $(CMAKE) $(MNN_HOME) $(TARGET_CMAKE_ARGS)
	$(CMAKE) --build $(TARGET_BUILD_DIR) -j $(NPROC) --target MNNV2Basic.out GetMNNInfo
	@touch $@

riscv: $(TARGET_STAMP)

$(CALIB_IMAGES):
	python3 $(APP_DIR)/scripts/gen-calib.py --out-dir $(CALIB_DIR) --count 8 --size 256 \
		--photos $(MNN_HOME)/resource/images/cat.jpg,$(MNN_HOME)/resource/images/TestMe.jpg

$(QUANT_CONFIG): $(APP_DIR)/quant-config.json.in $(CALIB_IMAGES)
	mkdir -p $(HOST_BUILD_DIR)
	sed 's|@CALIB_DIR@|$(abspath $(CALIB_DIR))|g' $< > $@

$(ONNX_FILE):
	@test -n "$(MOBILENET_V2_ONNX_URL)" -a -n "$(MOBILENET_V2_ONNX_SHA256)" || { \
		echo "error: supply ONNX_FILE and MOBILENET_V2_ONNX_SHA256, or a fixed MOBILENET_V2_ONNX_URL plus SHA256" >&2; exit 1; }
	mkdir -p $(dir $(ONNX_FILE))
	curl -fL --retry 3 -o "$@.download" "$(MOBILENET_V2_ONNX_URL)"
	@printf '%s  %s\n' "$(MOBILENET_V2_ONNX_SHA256)" "$@.download" | sha256sum -c -
	mv "$@.download" "$@"

model-source-ready: $(ONNX_FILE)
	@test -n "$(MOBILENET_V2_ONNX_SHA256)" || { echo "error: MOBILENET_V2_ONNX_SHA256 is required" >&2; exit 1; }
	@printf '%s  %s\n' "$(MOBILENET_V2_ONNX_SHA256)" "$(ONNX_FILE)" | sha256sum -c -

$(FP32_MODEL): $(HOST_CONVERT_BIN) $(ONNX_FILE) | model-source-ready
	$(HOST_CONVERT_BIN) -f ONNX --modelFile $(ONNX_FILE) --MNNModel $@ --bizCode xsai

fp32-model: $(FP32_MODEL)

$(FP16_MODEL): $(HOST_CONVERT_BIN) $(FP32_MODEL)
	$(HOST_CONVERT_BIN) -f MNN --modelFile $(FP32_MODEL) --MNNModel $@ --fp16

$(SQ_MODEL): $(HOST_QUANT_BIN) $(MODELS_DIR)/squeezenet-fp32.mnn $(QUANT_CONFIG)
	$(HOST_QUANT_BIN) $(MODELS_DIR)/squeezenet-fp32.mnn $@ $(QUANT_CONFIG)

$(INT8_MODEL): $(HOST_QUANT_BIN) $(FP32_MODEL) $(QUANT_CONFIG)
	$(HOST_QUANT_BIN) $(FP32_MODEL) $@ $(QUANT_CONFIG)

int8-model: $(INT8_MODEL)

model: $(INT8_MODEL)

install-binaries: $(TARGET_STAMP) $(TARGET_SIGNAL_BIN) $(TARGET_AMETEST_BIN)
	mkdir -p $(ROOTFS_BUILD_DIR)
	cp -f $(TARGET_V2BASIC_BIN) $(ROOTFS_V2BASIC_BIN)
	cp -f $(TARGET_GETINFO_BIN) $(ROOTFS_GETINFO_BIN)
	cp -f $(TARGET_SIGNAL_BIN) $(ROOTFS_SIGNAL_BIN)
	cp -f $(TARGET_AMETEST_BIN) $(ROOTFS_AMETEST_BIN)

# The MobileNet smoke is explicit; BF16/SqueezeNet experiment caches are not
# dependencies of the release install target.
install: install-binaries $(INT8_MODEL) $(INPUT_FILE)
	cp -f $(INT8_MODEL) $(ROOTFS_INT8_MODEL)
	cp -f $(INPUT_FILE) $(ROOTFS_INPUT)
	@ls -la $(ROOTFS_V2BASIC_BIN) $(ROOTFS_INT8_MODEL) $(ROOTFS_INPUT)

# Always regenerate: content is deterministic and generation is ~1s; avoids
# stale-file decisions on NFS where attribute caching can lie to make.
.PHONY: FORCE
FORCE:

$(INPUT_FILE): FORCE
	mkdir -p $(RUN_SMOKE_DIR)
	python3 $(APP_DIR)/scripts/gen-input.py --out $@ --size 224 \
		--photo $(MNN_HOME)/resource/images/cat.jpg

run-user: install $(INPUT_FILE)
	@mkdir -p $(RUN_SMOKE_DIR)/int8 && cp -f $(ROOTFS_V2BASIC_BIN) $(RUN_SMOKE_DIR)/int8/ && \
	cp -f $(INPUT_FILE) $(RUN_SMOKE_DIR)/int8/input_0.txt && \
	cd $(RUN_SMOKE_DIR)/int8 && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) $(QEMU_HOME)/build/qemu-riscv64 \
		-cpu $(QEMU_USER_CPU_FLAGS) ./MNNV2Basic.out $(V2BASIC_ARGS) && \
	test -s output.txt && echo "[mnn-run-user] output.txt: $(RUN_SMOKE_DIR)/int8/output.txt"

run-user-fp32: install $(FP32_MODEL) $(INPUT_FILE)
	@mkdir -p $(RUN_SMOKE_DIR)/fp32 && cp -f $(ROOTFS_V2BASIC_BIN) $(RUN_SMOKE_DIR)/fp32/ && \
	cp -f $(INPUT_FILE) $(RUN_SMOKE_DIR)/fp32/input_0.txt && \
	cd $(RUN_SMOKE_DIR)/fp32 && \
	QEMU_LD_PREFIX=$(QEMU_USER_PREFIX) $(QEMU_HOME)/build/qemu-riscv64 \
		-cpu $(QEMU_USER_CPU_FLAGS) ./MNNV2Basic.out \
		$(FP32_MODEL) 10 0 0 1 0 1x3x224x224 && \
	test -s output.txt && echo "[mnn-run-user] output.txt: $(RUN_SMOKE_DIR)/fp32/output.txt"

compare-fp32-int8: run-user run-user-fp32
	python3 $(APP_DIR)/scripts/compare-outputs.py \
		$(RUN_SMOKE_DIR)/int8/output.txt $(RUN_SMOKE_DIR)/fp32/output.txt

print-config:
	@echo "MNN_SRC       = $(MNN_SRC)"
	@echo "MNN_HOME      = $(MNN_HOME)"
	@echo "LLVM_HOME     = $(LLVM_HOME)"
	@echo "RISCV         = $(RISCV)"
	@echo "RISCV_SYSROOT = $(MNN_RISCV_SYSROOT)"
	@echo "HOST tools    = $(HOST_CONVERT_BIN) $(HOST_QUANT_BIN)"
	@echo "TARGET bin    = $(TARGET_V2BASIC_BIN)"

clean:
	rm -rf $(HOST_BUILD_DIR) $(TARGET_BUILD_DIR) $(APP_DIR)/build-smoke

distclean: clean
	rm -rf $(MODELS_DIR)
