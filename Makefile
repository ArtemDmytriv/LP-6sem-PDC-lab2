.SUFFIXES:
.SUFFIXES: .cpp .h .o

SRCDIR=src/
OBJDIR=obj/
INCDIR=include/

CXX=mpic++
CXXFLAGS+=-std=c++17 -I$(INCDIR)

SOURCES  = $(wildcard $(SRCDIR)*.cpp)
_OBJECTS = $(patsubst $(SRCDIR)%.cpp, %.o, $(SOURCES))
OBJECTS	= $(addprefix $(OBJDIR), $(_OBJECTS))

PROJ=lab2

all: $(PROJ)

$(PROJ): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

$(OBJECTS): $(OBJDIR)%.o: $(SRCDIR)%.cpp $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir $@

.PHONY: clean

clean:
	rm $(OBJDIR)*.o $(PROJ)
