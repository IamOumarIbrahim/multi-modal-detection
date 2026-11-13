#   MMSAR - late fusion of the two detector streams
#        Oumar Ibrahim U22200741


# s[n] = max(c_rgb[n], c_thermal[n])
def late_fusion(c_rgb, c_thermal):
    if len(c_rgb) != len(c_thermal):
        raise ValueError("rgb and thermal streams have different length")

    fused = []
    for i in range(len(c_rgb)):
        if c_rgb[i] >= c_thermal[i]:
            fused.append(c_rgb[i])
        else:
            fused.append(c_thermal[i])
    return fused
