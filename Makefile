#---------------------------------------------------------------------
# Makefile for VanitySearch
#
# Author : Jean-Luc PONS

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp hash/sha256_avx2.cpp hash/ripemd160_avx2.cpp \
      hash/sha256_avx512.cpp hash/ripemd160_avx512.cpp \
      Bech32.cpp Wildcard.cpp

OBJDIR = obj

ifdef gpu

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        hash/sha256_avx2.o hash/ripemd160_avx2.o \
        hash/sha256_avx512.o hash/ripemd160_avx512.o \
        GPU/GPUEngine.o Bech32.o Wildcard.o)

else

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o \
        hash/sha256_avx2.o hash/ripemd160_avx2.o \
        hash/sha256_avx512.o hash/ripemd160_avx512.o Bech32.o Wildcard.o)

endif

CXX        = g++
CUDA       = /usr/local/cuda-8.0
CXXCUDA    = /usr/bin/g++-4.8
NVCC       = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "5.2" -> "52"
ccap       = $(shell echo $(CCAP) | tr -d '.')

# Optimization flags
# Use "make portable=1" to build a binary that runs on any x86-64 CPU
# (runtime dispatch still enables SSE code paths only).
ifdef portable
OPTFLAGS   = -O3 -mssse3 -funroll-loops
else
OPTFLAGS   = -O3 -march=native -funroll-loops
endif

ifdef gpu
ifdef debug
CXXFLAGS   = -DWITHGPU -m64  -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   =  -DWITHGPU -m64 $(OPTFLAGS) -Wno-write-strings -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart
else
ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -g -I. -I$(CUDA)/include
else
CXXFLAGS   =  -m64 $(OPTFLAGS) -Wno-write-strings -I. -I$(CUDA)/include
endif
LFLAGS     = -lpthread
endif


#--------------------------------------------------------------------

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
endif

all: VanitySearch

# AVX2 hash kernels always need -mavx2 (they are gated behind a runtime
# CPUID check, so a portable build compiles them but only calls them on
# AVX2-capable CPUs).
$(OBJDIR)/hash/sha256_avx2.o : hash/sha256_avx2.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o $@ -c $<

$(OBJDIR)/hash/ripemd160_avx2.o : hash/ripemd160_avx2.cpp
	$(CXX) $(CXXFLAGS) -mavx2 -o $@ -c $<

$(OBJDIR)/hash/sha256_avx512.o : hash/sha256_avx512.cpp
	$(CXX) $(CXXFLAGS) -mavx512f -mavx512bw -o $@ -c $<

$(OBJDIR)/hash/ripemd160_avx512.o : hash/ripemd160_avx512.cpp
	$(CXX) $(CXXFLAGS) -mavx512f -mavx512bw -o $@ -c $<

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

VanitySearch: $(OBJET)
	@echo Making VanitySearch...
	$(CXX) $(OBJET) $(LFLAGS) -o VanitySearch

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o

