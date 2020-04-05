import numpy as np

def softmax(x): #hierarchichal softmax должен ускорить это дело, RP: миколов
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x) 
    return e_x / e_x.sum() 
   
class word2vec(object): 
    def __init__(self):
        self.N = 10
        self.X_train = [] 
        self.y_train = [] 
        self.window_size = 2
        self.alpha = 0.001
        self.words = [] 
        self.word_index = {} 
   
    def initialize(self,V,data): 
        self.V = V 
        self.W = np.random.uniform(-0.8, 0.8, (self.V, self.N)) 
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.V)) 
           
        self.words = data 
        for i in range(len(data)): 
            self.word_index[data[i]] = i 
   
    def init_with_given(self, word_index, W, W1):
        self.V = len(word_index)
        self.word_index = word_index.copy()
        self.words = list(word_index.keys())
        self.W = W.copy()
        self.W1 = W1.copy()
       
    def feed_forward(self,X): 
        self.h = np.dot(self.W.T,X).reshape(self.N,1)  
        self.u = np.dot(self.W1.T,self.h)
        #print(self.u) 
        self.y = softmax(self.u)   
        return self.y 
           
    def backpropagate(self,x,t): 
        e = self.y - np.asarray(t).reshape(self.V,1) 
        # e.shape is V x 1 
        dLdW1 = np.dot(self.h,e.T) 
        X = np.array(x).reshape(self.V,1) 
        dLdW = np.dot(X, np.dot(self.W1,e).T) 
        self.W1 = self.W1 - self.alpha*dLdW1 
        self.W = self.W - self.alpha*dLdW 
           
    def train(self,epochs): 
        for x in range(1,epochs):     
            print('current epoch: ', x)    
            self.loss = 0
            for j in range(len(self.X_train)): 
                self.feed_forward(self.X_train[j]) 
                self.backpropagate(self.X_train[j],self.y_train[j]) 
                C = 0
                for m in range(self.V): 
                    if(self.y_train[j][m]): 
                        self.loss += -1*self.u[m][0] 
                        C += 1
                self.loss += C*np.log(np.sum(np.exp(self.u))) 
            #print("epoch ",x, " loss = ",self.loss) 
            self.alpha *= 1/( (1+self.alpha*x) ) 
              
    def predict(self,word,number_of_predictions): 
        if word in self.word_index: 
            index = self.word_index[word] 
            X = [0 for i in range(self.V)] 
            X[index] = 1
            prediction = self.feed_forward(X) 
            output = {} 
            for i in range(self.V): 
                output[prediction[i][0]] = i 
               
            top_context_words = [] 
            for k in sorted(output,reverse=True): 
                top_context_words.append(self.words[output[k]]) 
                if(len(top_context_words)>=number_of_predictions): 
                    break

            return top_context_words 
        else: 
            return "Word not found in dicitonary" 

    def get_word_vector(self, word):
        if word in self.word_index:
            index = self.word_index[word] 
            X = [0 for i in range(self.V)] 
            X[index] = 1
            return np.dot(self.W.T,X).reshape(self.N,1).T  
        else:
            return "Word not found in dicitonary" 

    def save(self):
        w_filename = 'w_file.txt'
        w1_filename = 'w1_file.txt'
        word_filename = 'word_file.txt'

        with open(w_filename, 'w', encoding= 'utf-8') as f:
            for rw in self.W:
                row = [str(item) for item in rw]
                f.write(' '.join(row) + '\n')

        with open(w1_filename, 'w', encoding= 'utf-8') as f:
            for rw in self.W1:
                row = [str(item) for item in rw]
                f.write(' '.join(row) + '\n')

        with open(word_filename, 'w', encoding= 'utf-8') as f:
            for item in self.word_index.items():
                f.write(' '.join([str(i) for i in item]) + '\n')