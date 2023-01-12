#include "common.h"
#include <vector>
#include <cstring>

int main() {
	std::vector<MAT_PKT_T> lhs, rhs;
	COO_STREAM_T rst[NUM_PE];

	int matA[MAX_LENGTH][K], matB[K][MAX_LENGTH];
	int mat_final[MAX_LENGTH][MAX_LENGTH], mat_ref[MAX_LENGTH][MAX_LENGTH];

	memset(matA, 0, sizeof(matA));
	memset(matB, 0, sizeof(matB));
	memset(mat_final, 0, sizeof(mat_final));
	memset(mat_ref, 0, sizeof(mat_ref));

	printf("Prepare data ...");
	fflush(stdout);

	MAT_PKT_T pkt;
	for (int i=0; i<PACK_SIZE; i++)
		pkt.indices.data[i] = (NUM_NNZ+1) * (K/NUM_PE);
	lhs.push_back(pkt);
	rhs.push_back(pkt);

	for (int k=0; k<(K/NUM_PE); k++) {
		for (int j=0; j<PACK_SIZE; j++) {
			pkt.indices.data[j] = IDX_MARKER;
			pkt.vals.data[j] = NUM_NNZ;
		}
		lhs.push_back(pkt);
		rhs.push_back(pkt);
		for (int i=0; i<NUM_NNZ; i++) {
			for (int j=0; j<PACK_SIZE; j++) {
				pkt.indices.data[j] = (MAX_LENGTH / NUM_NNZ) * i;
				pkt.vals.data[j] = (i%4+1) * ((i&1) ? 1 : -1);
				matA[pkt.indices.data[j]][k*NUM_PE + j] = pkt.vals.data[j];
				matB[k*NUM_PE + j][pkt.indices.data[j]] = pkt.vals.data[j];
			}
			lhs.push_back(pkt);
			rhs.push_back(pkt);
		}
	}
	printf("done\n");
	fflush(stdout);

	spgemm(lhs.data(), rhs.data(),
			rst[0], rst[1], rst[2], rst[3],
			rst[4], rst[5], rst[6], rst[7]);

	printf("\n======================\n\n");

	printf("Calculate mat_ref\n");
	fflush(stdout);
	for (int i=0; i<MAX_LENGTH; i++)
		for (int j=0; j<MAX_LENGTH; j++) {
			int acc = 0;
			for (int k=0; k<K; k++) {
				acc += matA[i][k] * matB[k][j];
			}
			mat_ref[i][j] = acc;
		}

	printf("Collect rst to mat_final\n");
	fflush(stdout);
	for (int lane=0; lane<NUM_PE; lane++) {
		printf("\tPass rst[%d] of size %u\n", lane, rst[lane].size());
		bool exit = false;
		while (!exit) {
			COO_T coo = rst[lane].read();
			exit = (coo == END_COO);
			if (coo != END_COO) {
				mat_final[(unsigned)coo.row][(unsigned)coo.col] += coo.val;
				if (coo.row == 918) {
//					printf("{%u, %u, %d}\n", coo.row, coo.col, coo.val);
					fflush(stdout);
				}
			}
//			if (coo.val != mat_ref[coo.row][coo.col]) {
//				printf("COO{%u,%u,%d} of lane[%d] differs from mat_ref[%u][%u]=%d\n",
//						coo.row, coo.col, coo.val, lane, coo.row, coo.col, mat_ref[coo.row][coo.col]);
//			}
		}
	}

	printf("Compare mat_final against mat_ref\n");
	fflush(stdout);
	for (int i=0; i<MAX_LENGTH; i++)
		for (int j=0; j<MAX_LENGTH; j++) {
			if (mat_final[i][j] != mat_ref[i][j]) {
				printf("mat_final[%d][%d]=%d  !=  mat_ref[%d][%d]=%d\n",
						i, j, mat_final[i][j], i, j, mat_ref[i][j]);
				return 1;
			} else if (mat_final[i][j] != 0) {
//				printf("mat_final[%d][%d]=%d  ==  mat_ref[%d][%d]=%d\n",
//						i, j, mat_final[i][j], i, j, mat_ref[i][j]);
			}
		}

	return 0;

}
