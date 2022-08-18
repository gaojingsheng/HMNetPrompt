'''from summa.summarizer import summarize

origin_text = "the 20-year-old winger is due to play for psv eindhoven in thursday 's europa league qualifying tie against skn st polten but spurs are keen to get a deal wrapped up with a cash-only offer . video scroll down for wonderkid memphis depay takes a dip in the ocean on holiday . heading to the lane ? tottenham are keen to land dutch winger memphis depay . psv technical director marcel brands declared on wednesday the player was not for sale and that no bid had been received but sportsmail understands the offer has gone in and that psv are willing to accept . keen to strengthen : mauricio pochettino wants to add to his squad . depay was a success in the world cup for holland and was tipped as a target for louis van gaal at manchester united but he believes the player needs more time to develop . he scored 13 goals for psv last season . tottenham remain in negotiations with villarreal over a complicated deal for mateo musacchio with the spanish club keen for more cash to pay off third party owners river plate . by . jill reilly . published : . 06:28 est , 5 may 2012 . | . updated : . 06:32 est , 5 may 2012 . worrying news has emerged for homeowners looking to put their property on the market , as it has been revealed that house prices dropped by over # 900 a week last month . according to the halifax price index , the average home now costs # 159,883 , down 2.4 % per cent from # 163,796 in march - the same level as in 2004 . the ending of the stamp duty holiday . for first-time buyers in late march appears to have boosted home sales . early this year as buyers strove to beat the deadline , and has probably ."
origin_text = " by . jill reilly . published : . 06:28 est , 5 may 2012 . | . updated : . 06:32 est , 5 may 2012 . worrying news has emerged for homeowners looking to put their property on the market , as it has been revealed that house prices dropped by over # 900 a week last month . according to the halifax price index , the average home now costs # 159,883 , down 2.4 % per cent from # 163,796 in march - the same level as in 2004 . the ending of the stamp duty holiday . for first-time buyers in late march appears to have boosted home sales . early this year as buyers strove to beat the deadline , and has probably . contributed to the volatility in house prices , halifax said . decreasing value : according to the halifax price index , the average home now costs # 159,883 , down 2.4 % from # 163,796 in march . house prices have now fallen 11.8 per cent since they reached their peak in october 2007 . the figure is a uk average which does not take into account regional fluctuations . in april 2010 , as the property market hit a high point following its bounce back from the 2008/9 slump , the average price of a uk property was # 168,593 , halifax says . affordability : despite house price falls , uk property is still expensive compared to average earnings . the average home has lost # 10,000 since then . apr 11 # 160,785 may 11 # 161,039 jun 11 # 163,430 jul 11 # 163,765 aug 11 # 161,926 sep 11 # 161,368 oct 11 # 163,227 nov 11 # 161,556 dec 11 # 159,888 jan 12 # 160,925 feb 12 # 160,328 mar 12 # 163,796 apr 12 # 159,883 . source : halifax . but . last month 's 2.4 per cent fall follows an increase of 2.2 per cent in . march and actually leaves the more reliable three-month house price . change measure up 0.3 per cent , halifax said . that rise in prices in the three-month period was the first increase on a quarterly basis since september , following six successive falls . however , house prices have fallen 0.5 per cent over the past year , stand at the same level as in august 2004 and are down # 40,000 on the august 2007 peak . this week , nationwide 's house price index reported that property values last month saw a 0.2 per cent fall - the fourth month out of five that house prices have fallen . and nationwide warned that house prices , which on average were also 0.9 per cent down on april last year , were set to stagnate or fall throughout 2012 , as households ' confidence lagged behind any possible economic recovery . martin ellis , housing economist at halifax , said : ` despite the slight improvement in the underlying trend in recent months , house prices continue to lack real direction , with the current uk average price little different to where it was at the end of 2011 . '"
sum_text = summarize(origin_text, ratio = 0.5)
print('sum_lists:',words.keywords(origin_text))'''

import re
from Evaluation.OldROUGEEval import rouge_n

start_point=0
for k in range(len(turn["utt"]["word"])):
    if turn["utt"]["word"][k]==".":
        orign_list.append(turn["utt"]["word"][start_point:k+1])
        start_point=k+1
if turn["utt"]["word"]
        
        
origin_text = " ".join(turn["utt"]["word"])
orign_list=origin_text.split(origin_text)
for sum in data['summary']:
    sum_list=set()
    f1score=[]
    for i in range(len(orign_list)):
        f1score.append(rouge_n(orign_list[i],sum,n=1)(0))
    sum_list.append(orign_list.pop(f1score.index(max(f1score))))
        
