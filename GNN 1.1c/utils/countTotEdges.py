import sys,glob,string
edg=0
mydir=sys.argv[1]+"*"
#print mydir
for f in glob.glob(mydir):
    if string.find(f,'matrix')!=-1:
        edg+=file(f).read().count("1")      
print edg
