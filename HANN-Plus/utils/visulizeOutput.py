import io
import os


class WriteSentenceHeatmap(object):
    def __init__(self, savedir, num_of_reviews):
        super(WriteSentenceHeatmap, self).__init__()
        self.savedir = savedir + "/htmlhm"
        os.makedirs(self.savedir, exist_ok=True)

        for i in range(num_of_reviews):
            os.makedirs(self.savedir + "/{}".format(i), exist_ok=True)
            

    def js(self, attn_weight, input_index, voc_index2word, reviews_ctr, fname=''):
        with open(R'{}/{}.html'.format(self.savedir + "/{}".format(reviews_ctr), fname),'a') as file:
           text = "<!DOCTYPE html>\n<html>\n<body>\n<head>\n<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js'></script>\n</head><h1>review_{}</h1><div id='text_{}'>text goes here</div>\n".format(reviews_ctr, reviews_ctr)
           file.write(text)
           file.write('\n<script>\nvar words = [')

        for weight, index in zip(attn_weight, input_index):
            if(voc_index2word[index] != 'PAD'):
                with open(R'{}/{}.html'.format(self.savedir + "/{}".format(reviews_ctr), fname),'a') as file:
                    file.write("{{\n'word': '{}',".format(voc_index2word[index]))
                    file.write("'attention': {},\n}},".format(weight))
        
        with open(R'{}/{}.html'.format(self.savedir + "/{}".format(reviews_ctr), fname),'a') as file:
            file.write("];\n$('#text_")
            file.write("{}".format(reviews_ctr))
            file.write("').html($.map(words, function(w) {\nreturn '<span style=\"background-color:hsl(360,100%,' + (w.attention * -50 + 100) + '%)\">' + w.word + ' </span>'\n}))\n</script>")

    # # Write attention result into html file and txt filr.

    # def WriteAttention(self, USER_ATTN_RECORD):
    #     for userRecordObj in USER_ATTN_RECORD:
    #         # Create folder if not exists
    #         directory = ('AttentionVisualize/{}'.format(userRecordObj.userid))
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #         else:
    #             print('folder : {} exist'.format(directory))
            
    #         index_ = 0
    #         for sentence in userRecordObj.intra_attn_score:
    #             js( index_, sentence, directory)   
    #             index_ += 1  
            
    #         with open(R'{}/Inter_Attn_Score.txt'.format(directory),'a') as file:
    #             count = 0
    #             for score in userRecordObj.inter_attn_score:
    #                 file.write('Review {} : {}\n'.format(count, score))
    #                 count += 1
