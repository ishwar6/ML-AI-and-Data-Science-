def to_list(thing, transform  = lambda x: [x]):
    try: 
        return list(thing)
    except Exception as e:
        return transform(thing)



class Currency:
    def __init__(self,currency, value, multiplier):
        self.currency= currency
        self.value = value
        self.multiplier = multiplier

    def __eq__(self,other):
            try:
                ret = self.value*self.multiplier == other.value*other.multiplier
                return ret
            except:
                return False


print( Currency("EUR", 15, 0.5) == Currency("GBP", 10, 0.75) )


a = to_list(True, lambda x: [int (x)])
print(a)



