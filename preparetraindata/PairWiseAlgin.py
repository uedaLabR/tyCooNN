# Noguchi san's Piarwise mapper
from Bio import pairwise2
import multiprocessing
from multiprocessing import Pool
import tRex.genomemap.AbstractGenomeMap as AM
from tRex.TRexMappingResult import PairWiseResult
import time
from numba import jit, u1, i8, f8
from numba.types import types
import numpy as np


class PairWiseAlgin:

    def __init__(self, nameMatchMarge):
        self.nameMatchMarge = nameMatchMarge

    def genomeMap(self, reference_holder, reads, mapOption, MAXCORE):
        print('Mapping by Biopython pairwise2')
        self.reference_holder = reference_holder
        self.mapOption = mapOption
        start_time = time.time()
        # ref is RefHolder in TRexUtils
        ncore = multiprocessing.cpu_count()
        if ncore > MAXCORE:
            ncore = MAXCORE

        # with Pool(ncore) as p:
        #     mapped_reads = p.map(self.apply_pairwise_mapping, reads)
        mapped_reads = map(self.apply_pairwise_mapping, reads)

        mapped_reads = [read for read in mapped_reads if len(read.mapping_results) > 0]  # filtering.

        print('Finish. {}reads {}s\n'.format(len(mapped_reads), time.time() - start_time))
        return mapped_reads



    def apply_pairwise_mapping(self, read):

        score_and_species = []
        answer = read.inferencedtRNA
        keylist = self.reference_holder.getKeys()


        for species in keylist:

              if species.lower() == answer.lower():

                ref_seq = self.reference_holder.get_unmod_seq(species)
                score = pairwise2.align.localms(read.sequence, ref_seq, self.mapOption['match'],
                                                self.mapOption['mismatch'],
                                                self.mapOption['gapopen'], self.mapOption['gapex'], score_only=True)
                score_and_species.append([score, species])

        # obtain the start position and end position of alignment
        score_and_species = sorted(score_and_species, key=lambda x: x[0], reverse=True)
        number_of_hits = len(score_and_species)

        for i in range(number_of_hits):
            species = score_and_species[i][1]
            ref_unmod_seq = self.reference_holder.get_unmod_seq(species)
            alignment = pairwise2.align.localms(read.sequence, ref_unmod_seq, self.mapOption['match'],
                                                self.mapOption['mismatch'], self.mapOption['gapopen'],
                                                self.mapOption['gapex'])
            query_sequence = alignment[0].seqA
            ref_sequence = alignment[0].seqB

            r_st, r_en, q_st, q_en = get_start_and_end_index(query_sequence, ref_sequence, len(read.sequence),
                                                             len(ref_unmod_seq))

            result = PairWiseResult(tRNA_species=species, r_st=r_st, r_en=r_en,
                                    q_st=q_st, q_en=q_en, score=score_and_species[i][0], query_seq=query_sequence,
                                    ref_seq=ref_sequence)
            read.add_mapping_result(result)
        return read


@jit(i8[:](types.unicode_type, types.unicode_type, i8, i8), nopython=True)
def get_start_and_end_index(query_sequence, ref_sequence, query_length, ref_length):
    q_idx = -1
    r_idx = -1
    r_st = -1
    r_en = -1
    q_st = -1
    q_en = -1
    for n in range(len(query_sequence)):
        if query_sequence[n] != '-':
            q_idx += 1
        if ref_sequence[n] != '-':
            r_idx += 1
            if r_st == -1 and query_sequence[n] == ref_sequence[n]:
                # alignment start
                r_st = r_idx
                q_st = q_idx
        if q_idx + 1 == query_length or r_idx + 1 == ref_length:
            r_en = r_idx
            q_en = q_idx
            break
    return np.array([r_st, r_en, q_st, q_en], dtype=np.int64)
