import tensorflow as tf
import pickle
from p_util import minibatches, load_and_preprocess_data


class Config(object):

    n_features = 36
    n_classes = 3
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.002


class ParserModel(object):

    def add_placeholders(self):
       
        self.input_placeholder = tf.placeholder(tf.int32,(None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.float32,(None,self.config.n_classes))
        self.dropout_placeholder = tf.placeholder(tf.float32)
       
    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):      
        feed_dict = {}
        if(inputs_batch is not None):
            feed_dict[self.input_placeholder] = inputs_batch
        
        if(labels_batch is not None):
            feed_dict[self.labels_placeholder] = labels_batch
        
        if(dropout is not None):
            feed_dict[self.dropout_placeholder] = dropout
        
        return feed_dict

    def add_embedding(self):
      
        embedding_tensor = self.pretrained_embeddings
        #print('embedding_tensor',embedding_tensor.shape)
        embeddings = tf.nn.embedding_lookup(embedding_tensor,self.input_placeholder)
        #print('embedding shape',embeddings.shape)
        embeddings = tf.reshape(embeddings,(-1,self.config.n_features*self.config.embed_size))
        #print('embedding shape',embeddings.shape)
        return embeddings

    def add_prediction_op(self):     
        x = self.add_embedding()
        initializer = tf.contrib.layers.xavier_initializer()

        weights = tf.Variable(initializer([self.config.n_features*self.config.embed_size,self.config.hidden_size]),name='w')
        b1 = tf.Variable(tf.zeros(self.config.hidden_size),name='b1')
        h = tf.nn.relu(tf.matmul(x,weights) + b1)
        
        h_drop = tf.nn.dropout(h,keep_prob=1-self.config.dropout)

        Uts = tf.Variable(initializer([self.config.hidden_size, self.config.n_classes]),name='U')
        b2 = tf.Variable(tf.zeros(self.config.n_classes))
        pred = tf.matmul(h_drop,Uts) + b2
        
        return pred

    def add_loss_op(self, pred):       
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred))       
        return loss

    def add_training_op(self, loss):
        
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        return train_op

    def batch_train(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, parser, train_examples, dev_set):
        for i, (Xtr, Ytr) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.batch_train(sess, Xtr, Ytr)

        print ("Evaluating on dev set")
        dev_UAS, _ = parser.parse(dev_set)
        print ("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS

    def fit(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print ("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    saver.save(sess, './data/weights/parser.weights')

    def predict_on_batch(self, sess, inputs_batch):

        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()


def main(debug=True):
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)

    with tf.Graph().as_default():
        print("Started Building model")
        model = ParserModel(config, embeddings)
        parser.model = model
        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        with tf.Session() as session:
            parser.session = session
            session.run(init)
            print (50 * "-")
            print ("Training")
            model.fit(session, saver, parser, train_examples, dev_set)

            if not debug:
                print (50 * "-")
                print ("TESTING")
                print ("Restoring the best model weights found on the dev set")
                saver.restore(session, './data/weights/parser.weights')
                print ("Final evaluation on test set")
                UAS, dependencies = parser.parse(test_set)
                print ("- test UAS: {:.2f}".format(UAS * 100.0))
                print ("Writing predictions")
                with open('test.predicted.pkl', 'wb') as f:
                    pickle.dump(dependencies, f, -1)
                print ("Done!")

if __name__ == '__main__':
    main()


