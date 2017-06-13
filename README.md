# SUNRGBD

Python tool to curate the [SUNRGBD](http://rgbd.cs.princeton.edu/) database for semantic segmentation

Download the SUNRGBD database as well as the toolbox to get the labels:

```bash
wget http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
unzip SUNRGBD.zip
wget http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
unzip SUNRGBDtoolbox.zip
```

The parsing of the database is based on the `SunRGBDDataset` class which holds
the number of labels as a class attribute (`n_label=38`), and the relationship
between the label code (an integer between 0 and 37), and the label meaning ('lamp', 'bed', ...) in
the instance attribute `self.label_name`.

A curated database can be created using:

```python
python scripts/parse_database.py --path_2_database {path_2_database} --path_2_toolbox {path_2_toolbox} --output_dir {output_dir} --resize --nx 128 --ny 128 --grayscale
```

Which will create a database with grayscale images of size 128x128. It will
save the image in an `image` folder in `output_dir`, and the labels in a
`label` folder in `output_dir`. The database will be 170M.

# Example images

<img src="https://github.com/matthieule/sunrgbd/blob/master/data/0.png" alt="alt text" width=500px>

<img src="https://github.com/matthieule/sunrgbd/blob/master/data/100.png" alt="alt text" width=500px><Paste>

<img src="https://github.com/matthieule/sunrgbd/blob/master/data/600.png" alt="alt text" width=500px>

# References

[Song, S., Lichtenberg, S.P. and Xiao, J., 2015. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 567-576).](http://rgbd.cs.princeton.edu/paper.pdf)
