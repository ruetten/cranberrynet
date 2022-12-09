# cranberrynet

***When running, the nb files are Mathematica files, and the Output and Image folders need to go one folder back

# Problem
Frank Lazar is a close friend of Laik’s and works in the Cranberry Genetics and Genomics Laboratory (CGGL) under Prof. Zalapa on UW Madison campus, and is in need of a computer vision tool to quickly perform analysis on their cranberries. We were presented with two tasks. The first was to analyze images of full cranberries from different yields, recording data on individual size, width, height, and any other relevant data. The second was to analyze images of cut open cranberries, recording comparisons between the cranberries overall area to its hollow locule area.

# Motivation
With this tool, we hope to aid the researchers at the genetic labs by offering an easier and more efficient way to collect necessary data on their cranberries. Instead of having to measure, record, and analyze cranberry properties by hand, or paying a larger company to work on this, our tool only requires a loosely formatted image in order to present relevant statistics on the cranberries pictured. This is very useful to the researchers as it would save a lot of time and effort in their research efforts, helping the researchers in a very tangible way.

# Our Approach/Implementation
We aimed to reimplement existing solutions of detecting objects and finding their boundaries. The images that our tool receives has the cranberries spaced out in a grid-like formation with objects of known size also in the image. Therefore, the process of analyzing the cranberries involves identifying each cranberry and comparing them with the objects of known size. For the locule problem, we also had to differentiate the inner locule from the cranberry as a whole. We decided to use OpenCV, Python, and some Mathematica prototyping for our implementation, as it seemed simple to learn and effective for our needs.
