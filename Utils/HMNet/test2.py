import re

origin=["Coming", "up", "next", ":", "Breaking", "news", ":", "Republicans", "and", "Democrats", "reach", "a", "crucial", "budget", "deal", "tonight", ".", "Who", "won", "?", "Who", "lost", "?", "We", "have", "the", "details", "for", "you", ".", "Plus", ",", "a", "historic", "gathering", "of", "world", "leaders", "in", "South", "Africa", "today", ".", "But", "did", "President", "Obama", "'s", "handshake", "distract", "from", "Nelson", "Mandela", "'s", "memorial", "?", "And", "a", "family", "of", "six", "found", "alive", "after", "two", "days", "lost", "in", "the", "freezing", "wilderness", ".", "Great", "news", ".", "We", "'ll", "go", "to", "Nevada", "for", "the", "latest", "."]
def make_all_split_sentences(origin):
    origin_text = " ".join(origin).strip()
    print('origin:',origin_text[-1])
    # origin_list = origin_text.strip().split('.')
    #origin_list = list(filter(None, origin_text.strip().split('.')))
    origin_list = re.split(r'([.!?!.{6}])', origin_text)
    origin_list.append("")
    origin_list=["".join(i) for i in zip(origin_list[0::2],origin_list[1::2])]
    origin_list = list(filter(None, origin_list))
    origin_list = [origins.strip() for origins in origin_list]
    print('orign_list:',origin_list)
    return origin_list
origin_list=make_all_split_sentences(origin)