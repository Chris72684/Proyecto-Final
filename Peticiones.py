import Entrenamiento as Ent
while (True):
    annadir = input('Desea añadir usuario? ')
    if(annadir == 'Y' or annadir == 'y'):
        peti = Ent.pet()
        if (peti == True):
            print("Se guardaron las imagenes")
        else:
            print('Ha ocurrido un error. Programa finalizado')
            break
    else:
        break
if(peti == True):
    print('Dando comienzo a entenamiento...')
    entre = Ent.entrenamiento()
if (entre == True):
    Probar = input('Desea probar modelo? ')
if(Probar == 'Y' or Probar == 'y' and entre==True):
    Ent.Modelo()