#include "common.h"
#include <vector>
#include <cstring>

int main() {
	std::vector<MAT_PKT_T> lhs, rhs;
	COO_STREAM_T rst[NUM_PE];

	int matA[MAX_LENGTH][K], matB[K][MAX_LENGTH];

	memset(matA, 0, sizeof(matA));
	memset(matB, 0, sizeof(matB));
	const int NUM_NNZ = 10;

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
				pkt.vals.data[j] = i * ((i&1) ? 1 : -1);
				matA[pkt.indices.data[j]][k*NUM_PE + j] = pkt.vals.data[j];
				matB[k*NUM_PE + j][pkt.indices.data[j]] = pkt.vals.data[j];
			}
			lhs.push_back(pkt);
			rhs.push_back(pkt);
		}
	}
	printf("done\n");
	fflush(stdout);

	spgemm(lhs.data(), rhs.data(), rst);
	return 0;

}
