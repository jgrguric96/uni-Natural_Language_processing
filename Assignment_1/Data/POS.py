class POS:

    def __init__(self, finegrain, universal, occurrences=0, tokens=None):
            self.finegrain = finegrain
            self.universal = universal
            self.occurrences = occurrences
            self.tokens = tokens
            if tokens is None:
                self.tokens = dict()

    def add_token(self, token):
        if self.tokens.get(token):
            self.tokens[token] += 1
        else:
            self.tokens[token] = 1
