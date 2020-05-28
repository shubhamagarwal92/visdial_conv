[visdialconv_raw_amt_annotations.csv](visdialconv_raw_amt_annotations.csv) conatins the raw annotations provided by the AMT Turkers. 

We used the following mapping for amt annotations

```
        self.options_normalizer = {
            "hist_info": 1,
            "correctly": 2,
            "common_sense": 3,
            "guess": 4,
            "cant_tell": 5,
            "not_relevant": 6
        }
```


[crowdsourced_image_ids.txt](./crowdsourced_image_ids.txt) contains the image ids used for VisdialConv subset with verified dialog phenomena. As explained in the paper, we use only those responses for which more than 2 annotators agreed.  