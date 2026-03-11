#cuda destekli ekran kartı bende mps var
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS VAR")
else:
    device = torch.device("cpu")
    print("MPS YOK")
# 3*4 boyutunda bir matrisi zeros, ones ve rand metotlarıyla oluşturunuz.
matris_zeros = torch.zeros((3, 4), device=device)
matris_ones = torch.ones((3, 4), device=device)
matris_rand = torch.rand((3, 4), device=device)

print("\n--- Oluşturulan 3x4 Matrisler ---")
print("Zeros:\n", matris_zeros)
print("Ones:\n", matris_ones)
print("Rand:\n", matris_rand)

#  Bu matrislerin çarpımını cihaz üzerinde yapıp ekrana yazdırınız.
carpim_sonucu = torch.matmul(matris_ones, matris_rand.T)

print("\n--- Çarpım Sonucu (Ones x Rand.T) ---")
print(carpim_sonucu)