#include "common.h"

bool operator ==(const COO_T &a, const COO_T &b) {
	return (a.col == b.col) && (a.row == b.row) && (a.val == b.val);
}
bool operator !=(const COO_T &a, const COO_T &b) {
	return !((a.col == b.col) && (a.row == b.row) && (a.val == b.val));
}

template<typename T, unsigned len>
T array_max(T array[len]) {
    #pragma HLS inline
    #pragma HLS expression_balance
    T result = 0;
    for (unsigned i = 0; i < len; i++) {
        #pragma HLS unroll
        result = (array[i] > result)? array[i] : result;
    }
    return result;
}

#define LHS 0
#define RHS 1

template<unsigned side>
void loader(MAT_PKT_T *lhs, IDX_VAL_STREAM_T LD2PE[NUM_PE]) {
#pragma HLS INLINE off

#ifndef __SYNTHESIS__
	printf("loader%d\n", side);
	fflush(stdout);
#endif

	PACKED_IDX_T length_pkt = lhs[0].indices;
	IDX_T lengths[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lengths type=complete
	for (unsigned i=0; i<PACK_SIZE; i++)
#pragma HLS UNROLL
		lengths[i] = length_pkt.data[i];
	const unsigned max_length = array_max<IDX_T, PACK_SIZE>(lengths);

	for (unsigned i=0; i<max_length; i++) {
#pragma HLS PIPELINE II=1
		MAT_PKT_T pkt = lhs[i+1];

		for (unsigned j=0; j<PACK_SIZE; j++) {
#pragma HLS UNROLL
			if (i < lengths[j]) {
				if (pkt.indices.data[j] == IDX_MARKER) {
					IDX_VAL_T row_delta{IDX_MARKER, pkt.vals.data[j]};
					LD2PE[j].write(row_delta);
				} else {
					IDX_VAL_T col_val{pkt.indices.data[j], pkt.vals.data[j]};
					LD2PE[j].write(col_val);
				}
			}
		}
	}
}

// NOTE THAT: we assume every column of LHS and row of RHS has at least ONE elements.
void PE(IDX_VAL_STREAM_T &lhs_stream, IDX_VAL_STREAM_T &rhs_stream, COO_STREAM_T &dest) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
	printf("PE\n");
	fflush(stdout);
#endif

	for (int k=0; k < K/NUM_PE; k++) {
		IDX_VAL_T lhs_buff[BUFF_SIZE], rhs_buff[BUFF_SIZE];
#pragma HLS DATAFLOW
		const unsigned num_lhs_this_col = lhs_stream.read().val;
		const unsigned num_rhs_this_row = rhs_stream.read().val;
		for (int i=0; i<num_lhs_this_col; i++)
			lhs_buff[i] = lhs_stream.read();
		for (int i=0; i<num_rhs_this_row; i++)
			rhs_buff[i] = rhs_stream.read();

		for (int i = 0; i < num_lhs_this_col; i++) {
#pragma HLS LOOP_FLATTEN
			for (int j=0; j < num_rhs_this_row; j++) {
#pragma HLS PIPELINE II=1
				IDX_VAL_T lhs = lhs_buff[i], rhs = rhs_buff[j];
				COO_T rst{lhs.idx, rhs.idx, lhs.val * rhs.val};
				dest.write(rst);
			}
		}
	}
	dest.write(END_COO);
}

#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))));

template <int W>
inline void hashlookup3_core(ap_uint<W> key_val, ap_uint<64>& hash_val) {
    const int key96blen = W / 96;

    // key8blen is the BYTE len of the key.
    const int key8blen = W / 8;
    const ap_uint<32> c1 = 0xdeadbeef;
    //----------
    // body

    // use magic word(seed) to initial the output
    ap_uint<64> hash1 = 1032032634; // 0x3D83917A
    ap_uint<64> hash2 = 2818135537; // 0xA7F955F1

    // loop value 32 bit
    uint32_t a, b, c;
    a = b = c = c1 + ((ap_uint<32>)key8blen) + ((ap_uint<32>)hash1);
    c += (ap_uint<32>)hash2;


LOOP_lookup3_MAIN:
    for (int j = 0; j < key96blen; ++j) {
        a += key_val(96 * j + 31, 96 * j);
        b += key_val(96 * j + 63, 96 * j + 32);
        c += key_val(96 * j + 95, 96 * j + 64);

        a -= c;
        a ^= rot(c, 4);
        c += b;

        b -= a;
        b ^= rot(a, 6);
        a += c;

        c -= b;
        c ^= rot(b, 8);
        b += a;

        a -= c;
        a ^= rot(c, 16);
        c += b;

        b -= a;
        b ^= rot(a, 19);
        a += c;

        c -= b;
        c ^= rot(b, 4);
        b += a;
    }

    // tail	k8 is a temp
    // key8blen-12*key96blen will not large than 11
    switch (key8blen - 12 * key96blen) {
        case 12:
            c += key_val(W - 1, key96blen * 3 * 32 + 64);
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 11:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 10:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xffff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 9:
            c += key_val(W - 1, key96blen * 3 * 32 + 64) & 0xff;
            b += key_val(key96blen * 3 * 32 + 63, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 8:
            b += key_val(W - 1, key96blen * 3 * 32 + 32);
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 7:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 6:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xffff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;
        case 5:
            b += key_val(W - 1, key96blen * 3 * 32 + 32) & 0xff;
            a += key_val(key96blen * 3 * 32 + 31, key96blen * 3 * 32);
            break;

        case 4:
            a += key_val(W - 1, key96blen * 3 * 32);
            break;
        case 3:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffffff;
            break;
        case 2:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xffff;
            break;
        case 1:
            a += key_val(W - 1, key96blen * 3 * 32) & 0xff;
            break;

        default:
            break; // in the original algorithm case:0 will not appear
    }
    // finalization
    c ^= b;
    c -= rot(b, 14);

    a ^= c;
    a -= rot(c, 11);

    b ^= a;
    b -= rot(a, 25);

    c ^= b;
    c -= rot(b, 16);

    a ^= c;
    a -= rot(c, 4);

    b ^= a;
    b -= rot(a, 14);

    c ^= b;
    c -= rot(b, 24);

    hash1 = (ap_uint<64>)c;
    hash2 = (ap_uint<64>)b;

    hash_val = hash1 << 32 | hash2;
} // lookup3_64

inline unsigned own_hash(const unsigned row, const unsigned col, const int m=0) {
	int bits = m % (64/HASH_BIT);
	ap_uint<64> key_val, hash_val;
	key_val(31, 0) = row;
	key_val(63, 32) =col;

	hashlookup3_core(key_val, hash_val);
	return (unsigned)hash_val((bits+1)*HASH_BIT-1, bits*HASH_BIT);

}

class HashTable {
public:
	COO_STREAM_T in, out;
#pragma HLS STREAM variable=in depth=D
#pragma HLS STREAM variable=in depth=D
	COO_T ram[M][N];
#pragma HLS ARRAY_PARTITION variable=ram dim=1 type=complete

	void initialize() {
		for (int m=0; m<M; m++) {
#pragma HLS UNROLL
			for (int n=0; n<N; n++) {
#pragma HLS PIPELINE II=1
				ram[m][n] = EMPTY_COO;
			}
		}
	}

	void apply(COO_STREAM_T &dest, const int level, const int id) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
	printf("HashTable[%d][%d].apply()\n", level, id);
	fflush(stdout);
#endif


		initialize();
		bool exit = 0;
		while (!exit) {
#pragma HLS PIPELINE II=1
			COO_T coo = in.read();
			exit = (coo == END_COO);
			if (coo != END_COO) {
				unsigned row = coo.row;
				unsigned col = coo.col;
				int val = coo.val;

				bool drop = false, found=false;
				int sel, selH;
				COO_T dropped{(unsigned)-1, (unsigned)-1, INT_MAX};
				unsigned hash_val = own_hash(row, col);
				COO_T new_coo = (COO_T){row, col, val};
				for (int m=0; m<M; m++) {
#pragma HLS UNROLL
					COO_T old_coo = ram[m][hash_val];
					if (!found) {
						if (old_coo == EMPTY_COO) {
							found = true;
							sel = m;
							selH = hash_val;
						} else if (old_coo.col == col && old_coo.row == col) {
							found = true;
							sel = m;
							selH = hash_val;
							new_coo = (COO_T){row, col, old_coo.val + val};
						} else if (!drop || (abs(old_coo.val) < abs(dropped.val))) {
							drop = true;
							sel = m;
							selH = hash_val;
							dropped = old_coo;
						}
					}
				}
				if (!found) {
					assert(drop && "must drop if not found");
					dest.write(dropped);
				}
				ram[sel][selH] = new_coo;
			} else {
				// release all data
				for (int m=0; m<M; m++) {
					for (int h=0; h<N; h++) {
#pragma HLS PIPELINE II=1
						COO_T coo = ram[m][h];
						if (coo != EMPTY_COO) {
							dest.write(coo);
						}
					}
				}
				dest.write(END_COO);
			}
		}
	}
};

void arbiter_1p(
    const COO_T in_pld[num_lanes],
    COO_T resend_pld[num_lanes],
    const ap_uint<num_lanes> in_valid,
    ap_uint<1> in_resend[num_lanes],
    unsigned xbar_sel[num_lanes],
    ap_uint<num_lanes> &out_valid,
    const unsigned rotate_priority,
	const unsigned ref_bit
) {
    #pragma HLS pipeline II=1 enable_flush
    #pragma HLS latency min=ARBITER_LATENCY max=ARBITER_LATENCY

    #pragma HLS array_partition variable=in_addr complete
    #pragma HLS array_partition variable=xbar_sel complete

    // prioritized valid and addr
    ap_uint<num_lanes> arb_p_in_valid = in_valid;
    IDX_T arb_p_in_addr[num_lanes];
    IDX_T in_addr[num_lanes];
    #pragma HLS array_partition variable=in_addr complete
    #pragma HLS array_partition variable=arb_p_in_addr complete

    for (unsigned i = 0; i < num_lanes; i++) {
        #pragma HLS unroll
        arb_p_in_addr[i] = in_pld[(i + rotate_priority) % num_lanes].row;
        in_addr[i] = in_pld[i].row;
    }

    // array_rotate_left<IDX_T, num_lanes>(in_addr, arb_p_in_addr, rotate_priority);
    arb_p_in_valid.rrotate(rotate_priority);

    loop_A_arbsearch:
    for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
        #pragma HLS unroll
        bool found = false;
        unsigned chosen_port = 0;

        loop_ab_logic_encoder_unroll:
        for (unsigned ILid_plus_1 = num_lanes; ILid_plus_1 > 0; ILid_plus_1--) {
            #pragma HLS unroll
            if (arb_p_in_valid[ILid_plus_1 - 1] && (((arb_p_in_addr[ILid_plus_1 - 1] >> ref_bit) & 1) == OLid)) {
                chosen_port = ILid_plus_1 - 1;
                found = true;
            }
        }
        if (!found) {
            out_valid[OLid] = 0;
            xbar_sel[OLid] = 0;
        } else {
            out_valid[OLid] = 1;
            xbar_sel[OLid] = (chosen_port + rotate_priority) % num_lanes;
        }
    }

    loop_A_grant:
    for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
        #pragma HLS unroll
        unsigned requested_olid = in_addr[ILid] % num_lanes;
        bool in_granted = (in_valid[ILid]
                           && out_valid[requested_olid]
                           && (xbar_sel[requested_olid] == ILid));
        in_resend[ILid] = (in_valid[ILid] && !in_granted) ? 1 : 0;
        resend_pld[ILid] = in_pld[ILid];
    }

}

void shuffle_core(COO_STREAM_T input_lanes[2], COO_STREAM_T output_lanes[2], unsigned ref_bit) {
#pragma HLS INLINE off
	const unsigned shuffler_extra_iters = (ARBITER_LATENCY + 1) * num_lanes;
	// pipeline control variables
	ap_uint<num_lanes> fetch_complete = 0;
	unsigned loop_extra_iters = shuffler_extra_iters;
	ap_uint<1> state = SF_WORKING;
	bool loop_exit = false;

	// payloads
	COO_T payload[num_lanes];
	#pragma HLS array_partition variable=payload complete
	ap_uint<num_lanes> valid = 0;

	// resend control
	COO_T payload_resend[num_lanes];
	#pragma HLS array_partition variable=payload_resend complete

	ap_uint<1> resend[num_lanes];
	#pragma HLS array_partition variable=resend complete

	for (unsigned i = 0; i < num_lanes; i++) {
		#pragma HLS unroll
		resend[i] = 0;
	}

	// arbiter outputs
	unsigned xbar_sel[num_lanes];
	ap_uint<num_lanes> xbar_valid = 0;
	#pragma HLS array_partition variable=xbar_sel complete
	// arbiter priority rotation
	unsigned rotate_priority = 0;
	unsigned next_rotate_priority = 0;


	loop_shuffle_pipeline:
	while (!loop_exit) {
		#pragma HLS pipeline II=1
		#pragma HLS dependence variable=resend inter RAW true distance=6
		#pragma HLS dependence variable=payload_resend inter RAW true distance=6

		for (unsigned ILid = 0; ILid < num_lanes; ILid++) {
			#pragma HLS unroll
			if (resend[ILid]) {
				valid[ILid] = 1;
				payload[ILid] = payload_resend[ILid];
			} else if (fetch_complete[ILid]) {
				valid[ILid] = 0;
				payload[ILid] = END_COO;
			} else {
				if (input_lanes[ILid].read_nb(payload[ILid])) {
					if (payload[ILid] == END_COO) {
						fetch_complete[ILid] = 1;
						valid[ILid] = 0;
					} else {
						valid[ILid] = 1;
					}
				} else {
					valid[ILid] = 0;
					payload[ILid] = END_COO;
				}
			}
		}
		switch (state) {
		case SF_WORKING:
			if (fetch_complete.and_reduce()) {
				state = SF_ENDING;
			}
			break;
		case SF_ENDING:
			loop_extra_iters--;
			loop_exit = (loop_extra_iters == 0);
			break;
		default:
			break;
		}
		// ------- end of F stage

		// Arbiter stage (A) pipeline arbiter, depth = 6
		rotate_priority = next_rotate_priority;
		arbiter_1p(
			payload,
			payload_resend,
			valid,
			resend,
			xbar_sel,
			xbar_valid,
			rotate_priority,
			ref_bit
		);
		next_rotate_priority = (rotate_priority + 1) % num_lanes;
		// ------- end of A stage
		// crossbar stage (C)
		for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
			#pragma HLS unroll
			if (xbar_valid[OLid]) {
				if (valid[xbar_sel[OLid]]) {
					output_lanes[OLid].write(payload[xbar_sel[OLid]]);
				}
			}
		}
		// ------- end of C stage

	} // main while() loop ends here

	for (unsigned OLid = 0; OLid < num_lanes; OLid++) {
		#pragma HLS unroll
		output_lanes[OLid].write(END_COO);
	}
}

void collector(COO_STREAM_T &rst, COO_STREAM_T &pipe);

class Shuffler {
public:
	COO_STREAM_T in[2], out[2];
	void apply(COO_STREAM_T &dest0, COO_STREAM_T &dest1, unsigned ref_bit, const int level, const int id) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
	printf("Shuffler[%d[%d].apply()\n", level, id);
	fflush(stdout);
#endif

		#pragma HLS DATAFLOW
		shuffle_core(in, out, ref_bit);
		collector(out[0], dest0);
		collector(out[1], dest1);
	}
};


void collector(COO_STREAM_T &rst, COO_STREAM_T &pipe) {
	bool exit = false;
	while(!exit) {
#pragma HLS PIPELINE II=1
		COO_T coo = pipe.read();
		exit = (coo == END_COO);
		rst.write(coo);
	}
}

inline unsigned get_shuffle(int idx, int x) {
	return ((idx >> (x+1)) << x) | (idx & ((1<<x) - 1));
}
inline unsigned get_shuffle_channel(int idx, int x) {
	return (idx & (1 << (x-1)));
}

extern "C" {

void spgemm(MAT_PKT_T *lhs, MAT_PKT_T *rhs,  COO_STREAM_T rst[NUM_PE]) {

#pragma HLS interface m_axi offset=slave bundle=gmem0 port=lhs max_read_burst_length=64 num_read_outstanding=64 depth=MAX_LENGTH*K
#pragma HLS interface s_axilite bundle=control port=lhs
#pragma HLS interface m_axi offset=slave bundle=gmem1 port=rhs max_read_burst_length=64 num_read_outstanding=64 depth=MAX_LENGTH*K
#pragma HLS interface s_axilite bundle=control port=rhs
#pragma HLS interface axis port=rst

	IDX_VAL_STREAM_T LHS2PE[NUM_PE];
	IDX_VAL_STREAM_T RHS2PE[NUM_PE];

	HashTable hashes[NUM_HASH_LEVEL][NUM_PE];
	Shuffler shuffles[NUM_SHUFFLE_LEVEL][NUM_SHUFFLE_PER_LEVEL];

	loader<LHS>(lhs, LHS2PE);
	loader<RHS>(rhs, RHS2PE);

#pragma HLS DATAFLOW

	for (int i=0; i<NUM_PE; i++)
#pragma HLS UNROLL
		PE(LHS2PE[i], RHS2PE[i], hashes[0][i].in);

	for (int i=0; i<NUM_HASH_LEVEL-1; i++) {
#pragma HLS UNROLL
		for (int j=0; j<NUM_PE; j++) {
#pragma HLS UNROLL
			hashes[i][j].apply(shuffles[i][get_shuffle(j, i)].in[get_shuffle_channel(j, i)], i, j);
		}
		for (int j=0; j<NUM_SHUFFLE_PER_LEVEL; j++) {
#pragma HLS UNROLL
			shuffles[i][j].apply(hashes[i+1][j*2].in, hashes[i+1][j*2+1].in, MAX_LENGTH-j, i, j);
		}
	}

#ifndef __SYNTHESIS__
	printf("Start collectors\n");
	fflush(stdout);
#endif

	for (int j=0; j<NUM_PE; j++) {
#pragma HLS UNROLL
		collector(rst[j], hashes[NUM_HASH_LEVEL-1][j].out);
	}
}

}