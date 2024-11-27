import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import random

# Inicializa o mixer de áudio
pygame.mixer.init()

# Lista de áudios disponíveis
audio_files = ["bemtevi.mp3", "carcara.mp3", "joaodebarro.mp3", "sabia.mp3", "seriema.mp3"]

# Dicionário para mapear o nome dos arquivos para nomes descritivos
audio_nomes = {
    "bemtevi.mp3": "bem-te-vi",
    "carcara.mp3": "carcara",
    "joaodebarro.mp3": "joao-de-barro",
    "sabia.mp3": "sabia",
    "seriema.mp3": "seriema"
}

# Carrega as imagens para "ganhou" e "perdeu"
imagem_ganhou = cv2.imread("feliz.png", cv2.IMREAD_UNCHANGED)
imagem_perdeu = cv2.imread("triste.png", cv2.IMREAD_UNCHANGED)

# Carrega as imagens "sim" e "não"
imagem_sim = cv2.imread("sim.png")
imagem_nao = cv2.imread("nao.png")

# Redimensiona as imagens (ajuste os valores de largura e altura conforme necessário)
imagem_sim = cv2.resize(imagem_sim, (120, 55))
imagem_nao = cv2.resize(imagem_nao, (120, 55))

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Função para sobrepor imagem na testa
def sobrepor_imagem(frame, imagem, face, escala=1):
    # Calcula o centro da testa (média das coordenadas dos pontos de referência da testa)
    face = np.array([[coord.x, coord.y] for coord in face])
    pontos_testas = face[p_olhos]
    
    # Ajuste para posição mais alta
    centro_testas = np.mean(pontos_testas, axis=0) - [0.2, 0.3]  # Move a posição para cima
    
    # Posição da imagem na testa
    y, x = int(centro_testas[1] * frame.shape[0]), int(centro_testas[0] * frame.shape[1])

    # Ajusta o tamanho da imagem
    h, w, _ = imagem.shape
    h, w = int(h * escala), int(w * escala)
    imagem_redimensionada = cv2.resize(imagem, (w, h), interpolation=cv2.INTER_AREA)

    # Checando os limites da posição para evitar que a imagem saia da tela
    y = max(0, min(y, frame.shape[0] - h))
    x = max(0, min(x, frame.shape[1] - w))

    # Sobrepõe a imagem
    try:
        for c in range(3):  # Para cada canal de cor
            frame[y:y + h, x:x + w, c] = (imagem_redimensionada[:, :, c] *
                                          (imagem_redimensionada[:, :, 3] / 255.0) +
                                          frame[y:y + h, x:x + w, c] *
                                          (1.0 - imagem_redimensionada[:, :, 3] / 255.0))
    except Exception as e:
        print("Erro ao sobrepor imagem:", e)


# Lista de perguntas e respostas sobre sustentabilidade
perguntas_respostas = [
    ("Gosta de gastar muita agua?", "Não"),
    ("Gosta de usar sacolas reutilizaveis?", "Sim"),
    ("Gosta de jogar lixo na rua?", "Não"),
    ("Gosta de reutilizar papeis?", "Sim"),
    ("Gosta de poluir rios e lagos?", "Não"),
    ("Gosta de economizar energia?", "Sim"),
    ("Gosta de desmatar florestas?", "Não"),
    ("Gosta de reciclar latas de aluminio?", "Sim"),
    ("Gosta de usar pesticidas sem controle?", "Não"),
    ("Gosta de conservar a biodiversidade?", "Sim")
]


# variaveis pergunta aleatória e sua resposta
pergunta = ""
resposta = ""


# Limiares
ear_limiar = 0.27
mar_limiar = 0.35 
dormindo = 0


# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Inicializa o MediaPipe para Face Mesh e Mãos
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Estado do som
som_tocando = False
ultimo_tempo_audio = time.time()

# Estado do jogo
jogo_rodando = False
mao = "nada"
cont = 1
jogadas = 0

acertos = 0
erros = 0
margem = 0
reiniciar = True


with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh, mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.4) as hands:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue

        frame = cv2.flip(frame, 1)
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem com FaceMesh e Hands
        saida_facemesh = facemesh.process(frame)
        saida_hands = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        

        # Verifica se é hora de tocar um novo áudio
        if saida_facemesh.multi_face_landmarks:
            print("Rosto detectado")

            if not jogo_rodando:

                if jogadas == 0:
        
                    # gerando a pergunta aleatória
                    pergunta, resposta = random.choice(perguntas_respostas)

                    # Dimensões do retângulo preto
                    faixa_altura = 80  # Altura da faixa preta
                    faixa_cor = (0, 0, 0)  # Cor preta (BGR)

                    # Desenha o retângulo preto no topo do frame
                    frame[:faixa_altura, :] = faixa_cor

                    # Texto que será exibido sobre a faixa preta
                    texto = "Abra a boca para iniciar"
                    posicao = (130, 50)  # Posição do texto dentro da faixa
                    fonte = cv2.FONT_HERSHEY_DUPLEX
                    tamanho_fonte = 0.9
                    espessura = 2
                    cor_fonte = (255, 255, 255)  # Cor do texto (branca)

                    # Adiciona o texto na faixa preta
                    cv2.putText(frame, texto, posicao, fonte, tamanho_fonte, cor_fonte, espessura)

            
            tempo_atual = time.time()
            if tempo_atual - ultimo_tempo_audio >= 10 and som_tocando == False:
                audio_aleatorio = random.choice(audio_files)
                try:
                    pygame.mixer.music.load(audio_aleatorio)
                    pygame.mixer.music.play()
                    ultimo_tempo_audio = tempo_atual
                    nome_audio = audio_nomes.get(audio_aleatorio, "desconhecido")  # Obtém o nome descritivo do dicionário
                except pygame.error as e:
                    print(f"Erro ao carregar o áudio {audio_aleatorio}: {e}")

        else:
            print("Nenhum rosto detectado")
            pygame.mixer.music.stop()  # Para o som
            som_tocando = False  # Atualiza o estado para som parado
            jogo_rodando = False
            jogadas = 0
            acertos = 0
            erros = 0            

        if jogadas >= 1 and jogadas <= 3:
            jogo_rodando = True

        # Desenha os landmarks do rosto
        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                # Desenha os pontos do rosto
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

                # Chamada do EAR e MAR
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)           
                mar = calculo_mar(face, p_boca)

                # Verifica se a boca está aberta

                if jogadas == 0 and mar > mar_limiar and not jogo_rodando:
                    cv2.putText(frame, "Iniciando...", (150, 150),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1.5, (0, 0, 0), 2)
                    jogo_rodando = True

        except Exception as e:
            print("Erro:", e)


        # Detecção de mãos

        if jogo_rodando and saida_hands.multi_hand_landmarks:
            for hand_landmarks in saida_hands.multi_hand_landmarks:
                # Extrai as coordenadas do pulso (landmark 0) e das pontas dos dedos
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Verificar se a mão está apontada para cima (pulso abaixo das pontas dos dedos)
                if wrist.y > index_tip.y and wrist.y > pinky_tip.y:
                    # Determinar se é a mão direita ou esquerda
                    if wrist.x > 0.5:  # Coordenada x do pulso maior que 0.5
                        print("Mão Direita Apontada Para Cima")
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                        mao = "Não"
                    else:
                        print("Mão Esquerda Apontada Para Cima")
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        mao = "Sim"
                else:
                    print("Mão Não Está Apontada Para Cima")



        if jogo_rodando:
            # Exibindo a pergunta e a resposta
            print(f"Pergunta: {pergunta}")
            print(f"Resposta: {resposta}")

            # Centralizar a pergunta
            if len(pergunta) <= 27:
                margem = 110
            else:
                margem = 35

            print(margem)


            # Dimensões do retângulo preto
            faixa_altura = 80  # Altura da faixa preta
            faixa_cor = (0, 0, 0)  # Cor preta (BGR)

            # Desenha o retângulo preto no topo do frame
            frame[:faixa_altura, :] = faixa_cor

            # Texto que será exibido sobre a faixa preta
            posicao = (margem, 45)  # Posição do texto dentro da faixa
            fonte = cv2.FONT_HERSHEY_DUPLEX
            tamanho_fonte = 0.9
            espessura = 2
            cor_fonte = (255, 255, 255)  # Cor do texto (branca)

            # Adiciona o texto na faixa preta
            cv2.putText(frame, pergunta, posicao, fonte, tamanho_fonte, cor_fonte, espessura)

            if mao == "nada":
                print("continua")

                # Calcula a posição das imagens (ajuste as coordenadas conforme necessário)
                x_sim = 0
                y_sim = 240
                x_nao = largura - 120  # largura é a largura da imagem original
                y_nao = 240
                

                # Sobrepõe as imagens à imagem original
                frame[y_sim:y_sim+imagem_sim.shape[0], x_sim:x_sim+imagem_sim.shape[1]] = imagem_sim
                frame[y_nao:y_nao+imagem_nao.shape[0], x_nao:x_nao+imagem_nao.shape[1]] = imagem_nao

            else:
                if mao == resposta:
                    print("acertou")
                    # Exibe a imagem de "feliz"
                    sobrepor_imagem(frame, imagem_ganhou, face, escala=0.7)

                    if cont == 1:
                        acertos += 1
                        jogadas += 1

                    # Dimensões do retângulo preto
                    faixa_altura = 80  # Altura da faixa preta
                    faixa_cor = (0, 0, 0)  # Cor preta (BGR)

                    # Desenha o retângulo preto no topo do frame
                    frame[:faixa_altura, :] = faixa_cor

                    # Texto que será exibido sobre a faixa preta
                    texto = "Pergunta "+ str(jogadas) +"/3 - Acertos: "+ str(acertos) +" | Erros: "+ str(erros)
                    posicao = (40, 50)  # Posição do texto dentro da faixa
                    fonte = cv2.FONT_HERSHEY_DUPLEX
                    tamanho_fonte = 0.9
                    espessura = 2
                    cor_fonte = (255, 255, 255)  # Cor do texto (branca)

                    # Adiciona o texto na faixa preta
                    cv2.putText(frame, texto, posicao, fonte, tamanho_fonte, cor_fonte, espessura)

                    cont += 1

                else:
                    print("errou")
                    # Exibe a imagem de "triste"
                    sobrepor_imagem(frame, imagem_perdeu, face, escala=0.7)

                    if cont == 1:
                        erros += 1
                        jogadas += 1

                    # Dimensões do retângulo preto
                    faixa_altura = 80  # Altura da faixa preta
                    faixa_cor = (0, 0, 0)  # Cor preta (BGR)

                    # Desenha o retângulo preto no topo do frame
                    frame[:faixa_altura, :] = faixa_cor

                    # Texto que será exibido sobre a faixa preta
                    texto = "Pergunta "+ str(jogadas) +"/3 - Acertos: "+ str(acertos) +" | Erros: "+ str(erros)
                    posicao = (40, 50)  # Posição do texto dentro da faixa
                    fonte = cv2.FONT_HERSHEY_DUPLEX
                    tamanho_fonte = 0.9
                    espessura = 2
                    cor_fonte = (255, 255, 255)  # Cor do texto (branca)

                    # Adiciona o texto na faixa preta
                    cv2.putText(frame, texto, posicao, fonte, tamanho_fonte, cor_fonte, espessura)

                    cont += 1

        if cont >= 15: #determina o tempo para recomecar o jogo depois de uma resposta e resta as variavéis.
            cont = 1
            mao = "nada"
            jogo_rodando = False
            pergunta, resposta = random.choice(perguntas_respostas)                    

            if jogadas >= 3:

                while reiniciar:
                    if acertos > erros:
                        # Exibe a imagem de "triste"
                        sobrepor_imagem(frame, imagem_ganhou, face, escala=0.7)
                    else:
                        # Exibe a imagem de "triste"
                        sobrepor_imagem(frame, imagem_perdeu, face, escala=0.7)

                    if cv2.waitKey(10) & 0xFF == ord('a'):  
                        reiniciar = False
                    
                    if reiniciar == False :              
                        jogo_rodando = False
                        jogadas = 0
                        print("Iresultado")
                        print(acertos)
                        print(erros)
                        acertos = 0
                        erros = 0

                reiniciar = True
                    
            else:
                print("Eresultado")
                print(acertos)
                print(erros)
                print(jogadas)
            

        # Exibe o frame
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
