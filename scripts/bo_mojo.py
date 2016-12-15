from bs4 import BeautifulSoup
import urllib
import string
import unicodecsv as csv

listUrl="http://www.boxofficemojo.com/movies/alphabetical.htm" # Link for the lexicographically ordered list of movies
rootUrl='http://www.boxofficemojo.com/'

#alphabet contains the  values for the letter query parameter
alphabet=["NUM"]+list(string.ascii_uppercase) #NUM for the page containing movies not beginning with alphabets


#Extract three things for each row: the title of the movie, date of release and the link to page
with open ('movieLink.csv','wb') as csvFile:
    writer=csv.writer(csvFile,delimiter=',',encoding='utf-8')
    for letter in alphabet:
        for page in range(1,50):
            source=urllib.urlopen(listUrl+"?letter="+letter+"&page="+str(page)).read() #
            soup=BeautifulSoup(source,"lxml")
            soup=soup.find_all("table")[3]; #only contain the table which has the movies
            soup=soup.find_all("tr") #rows of the table, The first movie begins from index [1], index 0 contains the header row
            if len(soup)==1: #break if no movies on this page, move to the next letter
                break
            soup.pop(0)
            for movieRow in soup: #for each row in the collection of rows
                col=movieRow.findAll("td")
                movieTitle=col[0].text
                movieLink= col[0].a["href"] #just storing the suffix (after root url)
                movieDate= col[6].text
#                print movieTitle, movieDate, movieLink
                writer.writerow([movieTitle, movieDate, movieLink])
            
