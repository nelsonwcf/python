arr = ['p', 'e', 'r', 'f', 'e', 'c', 't', ' ', 
       'm', 'a', 'k', 'e', 's', ' ', 'p', 'r',
       'a', 'c', 't', 'i', 'c', 'e' ]

def reverse_words(arr):
    if len(arr) <= 1:
        return arr
    
    mirror_(arr)

    j = 0
    for i in range(len(arr) + 1):
        if i == len(arr) - 1 or arr[i] == ' ':
            mirror_(arr, j, i - 1)
            j = i + 1
    
    return arr

def mirror_(arr, i=0, j=len(arr) - 1):
    while i <= j:
        c = arr[i]
        arr[i] = arr[j]
        arr[j] = c
        i += 1
        j -= 1

reverse_words(arr)


























#def reverse_words(arr):
#  if arr is None or len(arr) == 1:
#    return arr
#    
#  _mirror(arr, 0, len(arr) - 1)
#    
#  start_position = 0
#    
#  for i in range(len(arr)):
#      if arr[i] is ' ':
#          end_position = i - 1
#          _mirror(arr, start_position, end_position)
#          start_position = i + 1
#            
#  _mirror(arr, start_position, len(arr) - 1)    
#    
#  return arr
#
#def _mirror(arr, start, end):
#  if start < end:
#      for i in range(1+(end - start)//2):
#          tmp = arr[start + i]
#          arr[start + i] = arr[end - i]
#          arr[end - i] = tmp