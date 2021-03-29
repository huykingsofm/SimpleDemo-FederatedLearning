import tenseal as ts
from tenseal.tensors.ckkstensor import CKKSTensor
import torch as th
from pprint import pprint
import numpy

matrix01 = th.tensor([[1, 2, 3]])
matrix02 = th.tensor([[4, 5, 6]])
print(matrix01, matrix02)

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

enc_matrix01 = ts.ckks_tensor(context, matrix01)
enc_matrix02 = ts.ckks_tensor(context, matrix02)
print(enc_matrix01, enc_matrix02)

print(enc_matrix01)

enc_matrix_sum: CKKSTensor = (enc_matrix01 + enc_matrix02) * 2
dec_matrix = enc_matrix_sum.decrypt()
pprint(dec_matrix.data)