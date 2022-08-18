import re

punctuation0 = '?!:(\\")'
punctuation1 = '(\\.{3})'
punctuation2 = '([a-z0-9]\.)'
punctuation3 ='(,\D)'
def double1(matched):
    #print(matched.group())
    punc = ' '+matched.group()+' '
    return punc
def double2(matched):
    #print(matched.group())
    punc = matched.group()[:-1]+' '+'.'
    return punc
origin_text ="bloomberg: \"protesters are taking jobs, away from the city\" . these are... the shocking! scenes that have? 20,000 U.S. dollar."
line1 = re.sub(r'[{}]+'.format(punctuation0), double1, origin_text)
print(line1)
line1_5 = re.sub(punctuation1, double1, line1)
print(line1_5)
line2 = re.sub(punctuation2, double2, line1)
print(line2)
line3 = re.sub(punctuation3, double1, line2)
print(line3)
res=list(filter(None,line3.split(' ')))
#ans=re.search('(\\.{3})','20,000 ,,U.S. s.b.')
#print(ans)
print(res)
