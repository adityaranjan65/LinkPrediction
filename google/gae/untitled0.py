def getShiftedString(s, leftShifts, rightShifts):
    # Write your code here
    s = ''.join([i for i in s if not i.isdigit()])
    print(s)
    Lfirst = s[0 : leftShifts] 
    Lsecond = s[leftShifts :] 
    new=Lsecond + Lfirst
    Rfirst = new[0 : len(new)-rightShifts]
    rightShifts = int(input().strip())

    result = getShiftedString(s, leftShifts, rightShifts)

    fptr.write(result + '\n')

    fptr.close()


    

s = input()
leftShifts = int(input().strip())
rightShifts = int(input().strip())
result = getShiftedString(s, leftShifts, rightShifts)
fptr.write(result + '\n')
fptr.close()
