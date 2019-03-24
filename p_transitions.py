class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence # just copies the list refrence
        self.stack = ["P_ROOT"]
        self.buffer = list(self.sentence) # actually copies the list
        self.dependencies = []


    def parse_step(self, transition):
        if (transition == "S" and len(self.buffer) != 0):
            self.stack.append(self.buffer.pop(0))
        elif ( transition == "LA" and len(self.stack) != 0):
            self.dependencies.append((self.stack[-1],self.stack.pop(-2)))
        elif (transition == "RA" and len(self.stack) != 0):
            self.dependencies.append((self.stack[-2],self.stack.pop(-1)))
        #print("stack values" ,self.stack)

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = list(partial_parses)
    while(unfinished_parses):
        minibatch = unfinished_parses[:batch_size]

        predict_transition = model.predict(minibatch)
        # print('transition list',predict_transition)

        for parse, transition in zip(minibatch, predict_transition):
            parse.parse_step(transition)
            if len(parse.stack) < 2 and len(parse.buffer) < 1:
                unfinished_parses.remove(parse)
    dependencies = [p.dependencies for p in partial_parses]
    return dependencies
