#Credits to Lin Zhouhan(@hantek) for the complete visualization code
import random, os, numpy, scipy
from codecs import open

class Visualization(object):
    def __init__(self, savedir, select_epoch, num_of_reviews):
        super(Visualization, self).__init__()
        self.savedir = savedir + "/VisualizeAttn/epoch_{}/htmlhm".format(select_epoch)
        os.makedirs(self.savedir, exist_ok=True)

        for i in range(num_of_reviews):
            os.makedirs(self.savedir + "/{}".format(i), exist_ok=True)
    
    def wdIndex2sentences(self, word_index, idx2wd, weights):
        
        words = [idx2wd[index] for index in word_index if idx2wd[index] != 'PAD']
        weights = [weights[index] for index, idx in enumerate(word_index) if idx2wd[idx] != 'PAD']
        sentence = [" ".join(words)]
        return sentence, [weights]

    def createHTML(self, texts, weights, reviews_ctr, fname):
        """
        Creates a html file with text heat.
        weights: attention weights for visualizing
        texts: text on which attention weights are to be visualized
        """
        
        fileName = R'{}/{}.html'.format(self.savedir + "/{}".
            format(reviews_ctr), fname)
        
        fOut = open(fileName, "w", encoding="utf-8")
        part1 = """
        <html lang="en">
        <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <style>
        body {
        font-family: Sans-Serif;
        }
        </style>
        </head>
        <body>
        <h3>
        Heatmaps
        </h3>
        </body>
        <script>
        """
        part2 = """
        var color = "255,0,0";
        var ngram_length = 3;
        var half_ngram = 1;
        for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
        intensity[i] = 0.0;
        // 新增規則 weights大於0才作畫
        for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
        if (i+j < intensity.length && i+j > -1 && trigram_weights[k][i] > 0) {
        intensity[i] += trigram_weights[k][i + j];
        }
        }
        if (i == 0 || i == intensity.length-1) {
        intensity[i] /= 2.0;
        } else {
        intensity[i] /= 3.0;
        }
        if (intensity[i] > max_intensity) {
        max_intensity = intensity[i];
        }
        if (intensity[i] < min_intensity) {
        min_intensity = intensity[i];
        }
        }
        var denominator = max_intensity - min_intensity;
        for (var i = 0; i < intensity.length; i++) {
        intensity[i] = (intensity[i] - min_intensity) / denominator;
        }
        if (k%2 == 0) {
        var heat_text = "<p><br><b>Example {fname}:</b><br>";
        } else {
        var heat_text = "<b>Example:</b><br>";
        }
        var space = "";
        for (var i = 0; i < tokens.length; i++) {
        heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
        if (space == "") {
        space = " ";
        }
        }
        //heat_text += "<p>";
        document.body.innerHTML += heat_text;
        }
        </script>
        </html>"""

        putQuote = lambda x: "\"%s\""%x
        textsString = "var any_text = [%s];\n"%(",".join(map(putQuote, texts)))
        weightsString = "var trigram_weights = [%s];\n"%(",".join(map(str,weights)))
        fOut.write(part1)
        fOut.write(textsString)
        fOut.write(weightsString)
        fOut.write(part2)
        fOut.close()
    
        return
