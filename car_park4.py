import cv2
import numpy as np
import json
import time


def load_parking_regions(json_file, video_width, video_height, original_width=1600, original_height=900):
    """Carregar e ajustar regiões de estacionamento a partir de um arquivo JSON para a resolução do vídeo."""
    with open(json_file, "r") as f:
        parking_regions = json.load(f)

    # Calcular os fatores de escala
    scale_x = video_width / original_width
    scale_y = video_height / original_height

    # Ajustar coordenadas de cada região
    for region in parking_regions:
        region["region"] = [
            [int(point[0] * scale_x), int(point[1] * scale_y)]
            for point in region["region"]
        ]
    return parking_regions


def detect_cars(frame, fg_bg_subtractor, min_area=500):
    """
    Detectar carros em movimento utilizando a subtração de fundo e contornos.
    """
    # Subtração de fundo
    fg_mask = fg_bg_subtractor.apply(frame)

    # Melhorar a máscara
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)

    # Detectar contornos
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    car_boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            car_boxes.append((x, y, x + w, y + h))

    return car_boxes


def count_parking_spots(video_path, output_video_path, json_file):
    """Identificar vagas de estacionamento disponíveis e contar carros ocupando as vagas."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Erro ao abrir o arquivo de vídeo"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Carregar e ajustar as regiões de vagas de estacionamento
    parking_regions = load_parking_regions(json_file, w, h)

    # Subtração de fundo para detecção de movimento
    fg_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # Dicionário para rastrear o tempo que os carros permanecem em uma vaga
    parked_cars = {}
    car_movement = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Fim do vídeo ou erro ao processar o quadro.")
            break

        # Detectar carros utilizando subtração de fundo
        car_boxes = detect_cars(frame, fg_bg_subtractor)

        # Avaliar as vagas
        cars_in_spots = 0
        for spot in parking_regions:
            region = spot["region"]
            is_occupied = False
            for box in car_boxes:
                # Ponto médio da caixa delimitadora
                box_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if cv2.pointPolygonTest(np.array(region, dtype=np.int32), box_center, False) >= 0:
                    car_id = tuple(map(int, box))  # Usar coordenadas como ID
                    if car_id not in parked_cars:
                        parked_cars[car_id] = time.time()  # Início do temporizador
                    elif time.time() - parked_cars[car_id] > 2:  # Carro está parado por mais de 2 segundos
                        is_occupied = True
                    break

            if is_occupied:
                cars_in_spots += 1
                color = (0, 0, 255)  # Vermelho para ocupadas
            else:
                color = (0, 255, 0)  # Verde para disponíveis
            cv2.polylines(frame, [np.array(region, dtype=np.int32)], isClosed=True, color=color, thickness=2)
            cv2.putText(frame, f"Spot {spot['id']} {'Occupied' if is_occupied else 'Available'}",
                        (region[0][0], region[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Contar carros que estão se movendo (entrando ou saindo)
        moving_cars = len([car_id for car_id, timer in parked_cars.items() if time.time() - timer < 2])

        # Exibir contagens de carros
        total_spots = len(parking_regions)
        available_spots = total_spots - cars_in_spots
        cv2.putText(frame, f"carros estacionados: {cars_in_spots}, Carros em movimento: {moving_cars}, vagas disponiveis: {available_spots}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        video_writer.write(frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

print("inicio_do_programa")

# Chamada da função com os parâmetros adequados
video_input = "C:\\Users\\samue\\Desktop\\Mestrado\\VideoEstacionamento\\video_diurno01.mp4"
count_parking_spots(video_input, "C:\\Users\\samue\\Desktop\\Mestrado\\VideoFinal\\video_diurno010.mp4", "C:\\Users\\samue\\Desktop\\Mestrado\\FrameExtraido\\frame_video_diurno01.json")