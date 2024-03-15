import random
import tornado.ioloop
import tornado.web
import tornado.websocket
import cv2
from picamera2 import Picamera2
import base64

from time import sleep


class WebSocketServer(tornado.websocket.WebSocketHandler):
    clients = set()

    def open(self):
        print("Client Connected")
        WebSocketServer.clients.add(self)

    def onClose(self):
        WebSocketServer.clients.remove(self)

    @classmethod
    def sendMessage(cls, message):
        for client in cls.clients:
            client.write_message(message)


class CameraData:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (640, 640)
        self.picam2.video_configuration.controls.FrameRate = 25.0
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.preview_configuration.align()
        self.picam2.start()

    # Grab an image from the pi camera, make it greyscale and convert it to a base64 string
    # Then send it over the websocket to the client
    def dataLoop(self):
        img = self.picam2.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, imgArray = cv2.imencode(".jpg", img)
        imgBytes = imgArray.tobytes()
        imgB64 = base64.b64encode(imgBytes)
        return imgB64


def main():
    app = tornado.web.Application(
        [("/websocket/", WebSocketServer)],
        websocket_ping_interval=10,
        websocket_ping_timeout=30,
    )
    app.listen(7890)

    ioLoop = tornado.ioloop.IOLoop.current()
    cameraData = CameraData()

    print("start")

    # Create a websocket server which constantly sends camera data to the client
    periodicCallback = tornado.ioloop.PeriodicCallback(
        lambda: WebSocketServer.sendMessage(cameraData.dataLoop()), 100
    )
    periodicCallback.start()

    ioLoop.start()


if __name__ == "__main__":
    main()

