import sys
from sdl2 import *
import ctypes
import numpy as np



class Pixart():
    def __init__(self, windowName=b"Pixart ML Window", windowSize=(800, 800), pixelResolution=(10, 10)):
        self.__width, self.__height = windowSize
        self.__matrix = np.zeros(pixelResolution, dtype=np.uint8())
        self.__pixWidth, self.__pixHeight = pixelResolution
        self.__window = SDL_CreateWindow(windowName, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                         self.__width, self.__height, SDL_WINDOW_SHOWN)
        self.__renderer = SDL_CreateRenderer(self.__window, -1, SDL_RENDERER_ACCELERATED)
        self.__thickness = 4
        self.__isMouseDown = False

    def getMatrix(self):
        return self.__matrix

    def run(self):
        SDL_Init(SDL_INIT_VIDEO)

        running = True
        while running:
            event = SDL_Event()
            SDL_SetRenderDrawColor(self.__renderer, 255, 255, 255, SDL_ALPHA_OPAQUE)
            SDL_RenderClear(self.__renderer)
            SDL_SetRenderDrawColor(self.__renderer, 0, 0, 0, SDL_ALPHA_OPAQUE)
            self.__drawGrid()
            self.__drawFilledSquares()
            SDL_RenderPresent(self.__renderer)

            while SDL_PollEvent(ctypes.byref(event)) != 0:
                if event.type == SDL_QUIT:
                    running = False
                    break
                if event.type == SDL_MOUSEBUTTONDOWN:
                    if event.button.button == SDL_BUTTON_LEFT:
                        self.__isMouseDown = True
                        self.__mouseMutateMatrix(event.button.x, event.button.y)
                if event.type == SDL_MOUSEBUTTONUP:
                    if event.button.button == SDL_BUTTON_LEFT:
                        self.__isMouseDown = False
                if event.type == SDL_MOUSEMOTION and self.__isMouseDown:
                    self.__mouseMutateMatrix(event.button.x, event.button.y)

        SDL_DestroyWindow(self.__window)
        SDL_DestroyRenderer(self.__renderer)
        SDL_Quit()

        # np.save('9', self.getMatrix())
        return 0

    def __drawGrid(self):
        rows, cols = self.__matrix.shape
        rect = SDL_FRect()
        rect.x = 0
        rect.w = self.__width
        rect.h = self.__thickness
        for i in range(rows + 1):
            rect.y = i * (self.__height - self.__thickness) / rows
            SDL_RenderFillRectF(self.__renderer, rect)

        rect.y = 0
        rect.h = self.__height
        rect.w = self.__thickness
        for i in range(cols + 1):
            rect.x = i * (self.__width - self.__thickness) / cols
            SDL_RenderFillRectF(self.__renderer, rect)

    def __drawFilledSquares(self):
        SDL_SetRenderDrawColor(self.__renderer, 255, 0, 0, SDL_ALPHA_OPAQUE)
        rows, cols = self.__matrix.shape
        rect = SDL_FRect()
        for y in range(rows):
            for x in range(cols):
                if self.__matrix[y, x] == 1:
                    rect.y = y * (self.__height - self.__thickness) / rows + self.__thickness / 2
                    rect.x = x * (self.__width - self.__thickness) / cols + self.__thickness / 2
                    rect.h = (self.__height - self.__thickness) / rows
                    rect.w = (self.__width - self.__thickness) / cols
                    SDL_RenderFillRectF(self.__renderer, rect)

    def __squareFinder(self, x, y):
        rows, cols = self.__matrix.shape
        i = (x - self.__thickness / 2) / ((self.__width - self.__thickness) / cols)
        j = (y - self.__thickness / 2) / ((self.__height - self.__thickness) / rows)
        return int(i), int(j)

    def __mouseMutateMatrix(self, x, y):
        i, j = self.__squareFinder(x, y)
        try:
            self.__matrix[j, i] = 1
        except IndexError: # the user might be able to select square outside the matrix
            pass


if __name__ == "__main__":
    pixart = Pixart(windowSize=(800, 800), pixelResolution=(50,50))
    sys.exit(pixart.run())
