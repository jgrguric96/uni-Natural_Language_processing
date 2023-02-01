import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy, time, datetime, random, gc, os
import torch
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences



class partC:

    def __init__(self):
        self.dataset = "SemEval2018-Task3/datasets/train/SemEval2018-T3-train-taskB_emoji.txt"
        self.nlp = spacy.load("en_core_web_sm")
        self.raw = None
        self.df = pd.DataFrame(columns=["Tweet Index", "Label", "Tweet Text"])
        if torch.cuda.is_available():

                # Tell PyTorch to use the GPU.
                self.device = torch.device("cuda")

                print('There are %d GPU(s) available.' % torch.cuda.device_count())

                print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def flat_f1(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return classification_report(labels_flat, pred_flat, output_dict=True)['macro avg']['f1-score']

    def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def start(self):
        df = pd.read_table(self.dataset, lineterminator="\n", sep="\t", quoting=3)
        print('Number of training sentences: {:,}\n'.format(df.shape[0]))

        df.columns = ['index', 'label', 'tweet']
        sentences = df.tweet.values
        labels = df.label.values
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                # max_length = 128,          # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )

            # Add the encoded sentence to the list.
            input_ids.append(encoded_sent)

        # Print sentence 0, now as a list of IDs.
        print('Original: ', sentences[0])
        print('Token IDs:', input_ids[0])

        # We'll borrow the `pad_sequences` utility function to do this.

        # Set the maximum sequence length.
        # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
        # maximum training sentence length of 47...
        MAX_LEN = 260

        print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                  value=0, truncating="post", padding="post")

        print('\nDone.')

        # Create attention masks
        attention_masks = []

        # For each sentence...
        for sent in input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)

        #Use 90% for training and 10% for validation
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                            test_size=0.1,
                                                                                            random_state=42)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                               test_size=0.1, random_state=42)
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        batch_size = 16

        print("Creating DataLoader for our training set...")
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        print("Done.")

        print("Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. ")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=4,  # The number of output labels--4 for multiclass classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        print(model.cuda())
        print("Done.")
        print("Creating AdamW optimizer")
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )
        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 4  # change to 5 for the plot

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []
        validation_loss = []
        torch.cuda.empty_cache()
        del [_, train_inputs, validation_inputs, train_labels, validation_labels]
        gc.collect()
        sumar = torch.cuda.memory_summary(device=self.device)
        print(sumar)
        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            torch.cuda.empty_cache()
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification

                torch.cuda.empty_cache()
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(self.format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy and f1-score for this batch of test sentences.
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                tmp_eval_f1 = self.flat_f1(logits, label_ids)

                # Accumulate the total accuracy and f1-score.
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1

                # Track the number of batches
                nb_eval_steps += 1

            validation_loss.append(eval_f1 / nb_eval_steps)
            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Macro-average F1: {0:.2f}".format(eval_f1 / nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))

        print("")
        print("Training complete!")

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(loss_values, 'b-o')

        # Label the plot.
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()

        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(validation_loss, 'b-o')

        # Label the plot.
        plt.title("Validation F1-score")
        plt.xlabel("Epoch")
        plt.ylabel("F1-score")

        plt.xticks(np.arange(5), ['1', '2', '3', '4', '5'])

        plt.show()

        df = pd.read_csv("SemEval2018-Task3/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt", delimiter='\t')
        df.columns = ['index', 'label', 'tweet']
        # Report the number of sentences.
        print('Number of test sentences: {:,}\n'.format(df.shape[0]))

        # Create sentence and label lists
        sentences = df.tweet.values
        labels = df.label.values

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            )

            input_ids.append(encoded_sent)

        # Pad our input tokens
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                                  dtype="long", truncating="post", padding="post")

        # Create attention masks
        attention_masks = []

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

            # Convert to tensors.
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)

        # Set the batch size.
        batch_size = 16

        # Create the DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

        # Put model in evaluation mode
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        print('    DONE.')

        pred_labels = []
        # Evaluate each test batch
        print('Calculating metrics for each batch...')

        # For each input batch...
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0"
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
            # Calculate and store the coef for this batch.
            pred_labels.append(pred_labels_i)

        matrix = confusion_matrix(np.array(true_labels).flatten(), np.array(pred_labels).flatten())
        report = classification_report(np.array(true_labels).flatten(), np.array(pred_labels).flatten())

        tn = []
        for i in range(len(matrix)):
            delrow = np.delete(matrix, i, 0)
            delcol = np.delete(delrow, i, 1)
            tn.append(delcol.sum())

        class_accuracies = (matrix.diagonal() + np.array(tn)) / matrix.sum()  # these should be good

        macro_avg_accuracy = class_accuracies.sum() / len(class_accuracies)  # idk about these
        weighted_avg_accuracy = balanced_accuracy_score(np.array(true_labels).flatten(),
                                                        np.array(pred_labels).flatten())

        print(matrix)
        print(class_accuracies)
        print(macro_avg_accuracy)
        print(weighted_avg_accuracy)
        print(report)

        output_dir = './model_save/'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    def analysis(self):
        print(
            "Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. ")
        MODEL_NAME = 'bert-base-uncased'
        MODEL_FINE = 'model_save'
        # model_base = BertForSequenceClassification.from_pretrained(
        #     MODEL_NAME,  # Use the 12-layer BERT model, with an uncased vocab.
        #     num_labels=4,  # The number of output labels--4 for multiclass classification.
        #     # You can increase this for multi-class tasks.
        #     output_attentions=False,  # Whether the model returns attentions weights.
        #     output_hidden_states=False,  # Whether the model returns all hidden-states.
        # )
        config_base = BertConfig.from_pretrained(MODEL_NAME, output_hidden_states=True, output_attentions=False, num_labels=4)
        model_base = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config_base)
        tokenizer_base = BertTokenizer.from_pretrained(MODEL_NAME)

        config_fine = BertConfig.from_pretrained(MODEL_FINE, output_hidden_states=True)
        model_fine = BertForSequenceClassification.from_pretrained(MODEL_FINE, config=config_base)
        tokenizer_fine = BertTokenizer.from_pretrained(MODEL_FINE)
        df = pd.read_table(self.dataset, lineterminator="\n", sep="\t", quoting=3)

        df.columns = ['index', 'label', 'tweet']
        sentences = df.tweet.values[:20]
        labels = df.label.values[:20]
        sentence_vectors = []
        sentence_vectors_fine = []
        all_tokens = []
        all_tokens_fine = []
        all_token_ids = []
        all_token_ids_fine = []

        for sentence in sentences:
            tokens = [tokenizer_base.cls_token] + tokenizer_base.tokenize(sentence) + [tokenizer_base.sep_token]
            tokens_fine = [tokenizer_fine.cls_token] + tokenizer_fine.tokenize(sentence) + [tokenizer_fine.sep_token]
            all_tokens.append(tokens)
            all_tokens_fine.append(tokens_fine)
            # print(tokens)

            token_ids = tokenizer_base.convert_tokens_to_ids(tokens)
            token_ids_fine = tokenizer_fine.convert_tokens_to_ids(tokens_fine)
            all_token_ids.append(token_ids)
            all_token_ids_fine.append(token_ids_fine)
            # print(token_ids)
            tokens_tensor = torch.tensor(token_ids).unsqueeze(0)
            tokens_tensor_fine = torch.tensor(token_ids_fine).unsqueeze(0)

            model_base.eval()  # turn off dropout layers
            model_fine.eval()
            output = model_base(tokens_tensor)
            output_fine = model_fine(tokens_tensor_fine)

            layers = output.hidden_states
            layers_fine = output_fine.hidden_states

            # Our batch consists of a single sentence, so we simply extract the first one
            sentence_vector = layers[len(layers)-1][0].detach().numpy()
            sentence_vector_fine = layers_fine[len(layers_fine) - 1][0].detach().numpy()

            # The sentence vector is a list of vectors, one for each token in the sentence
            # Each token vector consists of 768 dimensions
            # print(sentence_vector.shape)

            # We use the vector for the first token (the CLS token) as representation for the sentence
            sentence_vectors.append(sentence_vector[0])
            sentence_vectors_fine.append(sentence_vector_fine[0])

        similarity_matrix = cosine_similarity(np.asarray(sentence_vectors))
        similarity_matrix_fine = cosine_similarity(np.asarray(sentence_vectors_fine))
        # print(similarity_matrix)

        # Plot a heatmap
        ax = sns.heatmap(similarity_matrix, linewidth=0.5, cmap="YlGnBu", annot=True, annot_kws={"size": 5})
        ids = list(range(1, len(sentences) + 1))
        ax.set_xticklabels(ids)
        ax.set_yticklabels(ids)

        # Remove the ticks, but keep the labels
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labeltop=True,
                       labelbottom=False)
        ax.set_title("Similarity between sentence pairs")
        plt.show()

        ax_fine = sns.heatmap(similarity_matrix_fine, linewidth=0.5, cmap="YlGnBu", annot=True, annot_kws={"size": 5})
        ax_fine.set_xticklabels(ids)
        ax_fine.set_yticklabels(ids)


        ax_fine.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labeltop=True,
                       labelbottom=False)
        ax_fine.set_title("Similarity between sentence pairs for fine tuned BERT")
        plt.show()


if __name__ == '__main__':
    subtaskC = partC()
    subtaskC.analysis()
    subtaskC.start()
