APP = Konane
SRC = Project2.py

all: $(APP)

$(APP): $(SRC)
	dos2unix $(SRC)
	cp $(SRC) $(APP)
	chmod +x $(APP)

clean:
	rm -f $(APP)
	rm -rf __pycache__ *.pyc