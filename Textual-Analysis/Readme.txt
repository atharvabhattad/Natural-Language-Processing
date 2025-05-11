My approach to solve the problem statement

1) First step was web Scrapping using python library bs4. It included 
   surfing some of the datasets randomly to find where the heading and 
   text is present in site.This was done using inspect tool provided 
   by all browsers. 

2) All the stopwords files were converted into a single file called 
   combined_files.text

3) Then stopwords were removed from extracted text

4) Text was converted into paragragh

5) Lemmatization was done on Text

6) Stopwords were removed from Master_dictionary 

7) All the parameters of textual analysis were defined

8) Finally the dataframe recieved here was concanated with the input 
   file and Final output file was produced.


How to run the .py file to generate output

=> To run the .ipynb file just change the all paths used in code according 
   to local system and internet connectivity is required while running code.



Important:
Some of the of links were unable to be web scraped due to limitation of library.
(Also the ways of including class were used to get to conclusion)