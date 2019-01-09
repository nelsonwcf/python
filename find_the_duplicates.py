def find_duplicates(arr1, arr2):
    out = []
    i1 = 0
    i2 = 0
    
    while i1 < len(arr1) and i2 < len(arr2):
        if arr1[i1] == arr2[i2]:
            out.append(arr1[i1])
            i1 += 1
            i2 += 1            
        elif arr1[i1] > arr2[i2]:
            i2 += 1
        else:
            i1 += 1

    return out