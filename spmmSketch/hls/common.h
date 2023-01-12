#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <limits.h>
#include <assert.h>

const int NUM_NNZ = 10;
const int MAX_LENGTH=1024;
const int HIGH_BIT=9; // log(MAX_LENGTH)
const int K=128;

const int BUFF_SIZE=128; // max num of elements per col/row of lhs/rhs

const int NUM_PE=8;
const int NUM_PE_BITS=3; // Log2(NUM_PE)
const int NUM_HASH_LEVEL=4; // Log2(NUM_PE) + 1
const int NUM_SHUFFLE_LEVEL=3; // Log2(NUM_PE)
const int NUM_SHUFFLE_PER_LEVEL=NUM_PE/2;
const int PACK_SIZE=NUM_PE;

const int M=4;
const int N=256;
const int HASH_BIT=7; // log2(K)
const int D=16;

const int THRESHOLD=150;

const unsigned ARBITER_LATENCY = 5;
const unsigned num_lanes = 2;

typedef unsigned IDX_T;
#define IDX_MARKER 0xffffffff
typedef int VAL_T;

#define SF_WORKING 1
#define SF_ENDING 0


typedef struct {IDX_T data[PACK_SIZE];} PACKED_IDX_T;
typedef struct {VAL_T data[PACK_SIZE];} PACKED_VAL_T;

typedef struct {
	PACKED_IDX_T indices;
	PACKED_VAL_T vals;
} MAT_PKT_T;

typedef struct {
	IDX_T idx;
	VAL_T val;
} IDX_VAL_T;

typedef struct {
	IDX_T row;
	IDX_T col;
	VAL_T val;
} COO_T;

const COO_T EMPTY_COO{(unsigned)-1,(unsigned)-1, -2};
const COO_T END_COO{(unsigned)-1,(unsigned)-1,-1};

bool operator ==(const COO_T &a, const COO_T &b);
bool operator !=(const COO_T &a, const COO_T &b);

typedef hls::stream<IDX_VAL_T> IDX_VAL_STREAM_T;
typedef hls::stream<COO_T> COO_STREAM_T;

extern "C" {
void spgemm(MAT_PKT_T *lhs, MAT_PKT_T *rhs,
		COO_STREAM_T &rst0, COO_STREAM_T &rst1, COO_STREAM_T &rst2, COO_STREAM_T &rst3,
		COO_STREAM_T &rst4, COO_STREAM_T &rst5, COO_STREAM_T &rst6, COO_STREAM_T &rst7);
}
