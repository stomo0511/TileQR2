#
UNAME = $(shell uname)
ifeq ($(UNAME),Linux)
  BLAS_ROOT = /opt/intel/compilers_and_libraries/linux/mkl
  BLAS_INC_DIR = $(BLAS_ROOT)/include
  BLAS_LIB_DIR = $(BLAS_ROOT)/lib/intel64
  SBLAS_LIBS = -Wl,--start-group $(BLAS_LIB_DIR)/libmkl_intel_lp64.a $(BLAS_LIB_DIR)/libmkl_sequential.a $(BLAS_LIB_DIR)/libmkl_core.a -Wl,--end-group -lpthread -ldl -lm
  CXX =	g++
endif
ifeq ($(UNAME),Darwin)
  BLAS_ROOT = /opt/intel/compilers_and_libraries/mac/mkl
  BLAS_INC_DIR = $(BLAS_ROOT)/include
  BLAS_LIB_DIR = $(BLAS_ROOT)/lib
  SBLAS_LIBS = $(BLAS_LIB_DIR)/libmkl_intel_lp64.a $(BLAS_LIB_DIR)/libmkl_sequential.a $(BLAS_LIB_DIR)/libmkl_core.a -lpthread -ldl -lm
  CXX =	/usr/local/bin/g++ 
endif

#
PLASMA_ROOT = /opt/plasma-17.1
PLASMA_INC_DIR = $(PLASMA_ROOT)/include
PLASMA_LIB_DIR = $(PLASMA_ROOT)/lib
PLASMA_LIBS = -lcoreblas -lplasma
#
CXXFLAGS =	-fopenmp -m64 -O2 -I$(BLAS_INC_DIR) -I$(PLASMA_INC_DIR)

RLOBJS =	TileMatrix.o TileQR.o Check_Accuracy.o RightLooking.o
RTOBJS =	TileMatrix.o TileQR.o Check_Accuracy.o RightLooking_Task.o

all:	RL RT

TD:	$(TDOBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(TDOBJS) \
				-L$(PLASMA_LIB_DIR) $(PLASMA_LIBS) \
				$(SBLAS_LIBS)
				
RL:	$(RLOBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(RLOBJS) \
				-L$(PLASMA_LIB_DIR) $(PLASMA_LIBS) \
				$(SBLAS_LIBS)				

RT:	$(RTOBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(RTOBJS) \
				-L$(PLASMA_LIB_DIR) $(PLASMA_LIBS) \
				$(SBLAS_LIBS)

clean:
	rm -f $(RLOBJS)　$(RTOBJS)
