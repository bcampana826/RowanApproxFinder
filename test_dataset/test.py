
def toy_world():
    inp = input('number of verts: ')

    f = open("toy-world.txt", "w")
    f.write("t 1 "+str(inp)+"\n")

    for i in range(int(inp)):
        f.write("v "+ str(i)+ " "+str(i)+"\n")

    for i in range(int(inp)):
        for j in range(int(inp)):   
            if i != j:
                f.write("e "+ str(i)+ " "+ str(j)+"\n")

    f.close()

def read_files():
    label = input('label file path: ')
    edgelist = input('edgelist file path: ')
    output = input('output file: ')

    with open(label) as l:
        labels = [line.strip() for line in l]

    with open(edgelist) as e:
        edges = [line.strip() for line in e]

    print(edges)

    f = open(output, "w")
    
    f.write("t 1 "+str(len(labels))+"\n")

    for i in range(len(labels)):
        sp = labels[i].split()

        f.write("v "+str(sp[0])+" "+str(sp[1])+"\n")

    for j in range(len(edges)):
        sp = edges[j].split()
        
        if sp[0] != sp[1]:
            f.write("e "+str(sp[0])+" "+str(sp[1])+"\n")

    f.close()



val = input("1 for toy, 2 for real: ")
if val == "1":
    toy_world()
elif val == "2":
    read_files()