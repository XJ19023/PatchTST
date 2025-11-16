import torch
import math
def quadratic(a, b, c):
    d = b**2 - 4*a*c

    if d < 0:
        print("无解")
    else:
        e = math.sqrt(d)
        x1 = (-b + e) / (2 * a)
        x2 = (-b - e) / (2 * a)
        print("x1 =", x1, "\t", "x2 =", x2)
if False:
    spark_pe = 296.69
    spark_decoder = 6.42
    spark_encoder = 20.328
    spark = spark_pe * 4096 + spark_decoder * 128 + spark_encoder * 64
    print(f"Spark Encoder: {spark_encoder * 64:.2f}")
    print(f"Spark Decoder: {spark_decoder * 128:.2f}")
    print(f"Spark: {spark * 10**(-6):.3f}")

    olive_pe = 492.74
    ant_decoder = 14.784
    ant_encoder = 18.648
    ant = olive_pe * 4096 + ant_decoder * 128 + ant_encoder * 64
    print(f"Ant Encoder: {ant_encoder * 64:.2f}")
    print(f"Ant Decoder: {ant_decoder * 128:.2f}")
    print(f"Ant: {ant * 10**(-6):.3f}")

    olive_decoder_4 = 48.51
    olive_decoder_8 = 73.25
    olive_decoder = olive_decoder_4
    olive_encoder = 61.488
    olive = olive_pe * 4096 + olive_decoder * 128 + olive_encoder * 64
    print(f"Olive Encoder: {olive_encoder * 64:.2f}")
    print(f"Olive Decoder: {olive_decoder * 128:.2f}")
    print(f"Olive: {olive * 10**(-6):.3f}")

    mant_pe = 540.96
    mant = mant_pe * 4096
    print(f"Mant: {mant * 10**(-6):.3f}")

    WARD_PE = 312.31
    ward_decoder = 1500.4
    ward = WARD_PE * 4096 + ward_decoder
    print(f"Ward: {ward * 10**(-6):.3f}")

    print('='*50)
    print('Spark: ', end='')
    quadratic(spark_pe, (spark_encoder+2*spark_decoder), -ward)
    print('ANT: ', end='')
    quadratic(olive_pe, (ant_encoder+2*ant_decoder), -ward)
    print('Olive: ', end='')
    quadratic(olive_pe, (olive_encoder+2*olive_decoder), -ward)
    print('Mant: ', end='')
    quadratic(mant_pe, 0, -ward)

if True:
    spark_pe = torch.tensor([2001, 189150])
    spark_decoder = torch.tensor([41.726, 7074]) * 128
    spark_encoder = torch.tensor([157.46, 14241]) * 64
    spark_decoder = torch.tensor([41.726, 7074]) * 1
    spark_encoder = torch.tensor([157.46, 14241]) * 1
    # print(f"Spark PE: {spark_pe}")
    # print(f"Spark De: {spark_decoder}")
    # print(f"Spark En: {spark_encoder}")
    print(f"Spark: {spark_encoder + spark_decoder}")

    ant_pe = [3474, 306780]
    ant_decoder = torch.tensor([97.94, 13099]) * 128
    ant_encoder = torch.tensor([155.8, 9876]) * 64
    ant_decoder = torch.tensor([97.94, 13099]) * 1
    ant_encoder = torch.tensor([155.8, 9876]) * 1
    # print(f"ANT PE: {ant_pe}")
    # print(f"ANT De: {ant_decoder}")
    # print(f"ANT En: {ant_encoder}")
    print(f"ANT: {ant_encoder + ant_decoder}")


    olive_decoder = torch.tensor([297.94, 53099]) * 128
    olive_encoder = torch.tensor([422.728, 30409]) * 64
    olive_decoder = torch.tensor([297.94, 53099]) * 1
    olive_encoder = torch.tensor([422.728, 30409]) * 1
    # print(f"OliVe De: {olive_decoder}")
    # print(f"OliVe En: {olive_encoder}")
    print(f"OliVe: {olive_encoder + olive_decoder}")

    mant_pe = [4022, 480587]
    # print(f"Mant: {mant_pe}")

    WARD_PE = [2122, 196529]
    ward_encoder = torch.tensor([14612, 1836900])
    ward_encoder = torch.tensor([14612, 1836900]) / 64
    # print(f"Ward PE: {WARD_PE}")
    print(f"Ward En: {ward_encoder}")
