'''
This file implements a simple user interface to experiment with the Hopfield network.
'''

import sys
from sdl2 import *
import ctypes
import numpy as np
from typing import Tuple


class Pixart():
    """
    This class creates a user tool the create a binary image
    that can be tested in the Hopfield network.
    """
    def __init__(self: Pixart,
                 windowName: str = b"Pixart ML Window",
                 windowSize: Tuple[int, int] = (800, 800),
                 pixelResolution: Tuple[int, int] = (31, 31),
                 matrix: np.ndarray = None
    ):
        """
        SUMMARY
            Constructor of the Pixart class.
        PARAMETERS
            windowName str = b"Pixart ML Window": the name of the window that is being created,
            windowSize Tuple[int, int] = (800, 800): tuple containing the dimensions of the window,
            pixelResolution Tuple[int, int] = (31, 31): the number of rows and columns in the window,
            matrix np.ndarray = None: the matrix of zeros and ones to which the window is initialised
        RETURN
            None
        """
        self.__width, self.__height = windowSize
        if matrix is None:
            self.__matrix = np.zeros(pixelResolution, dtype=np.uint8())
        else:
            self.__matrix = matrix
        self.__pixWidth, self.__pixHeight = pixelResolution
        self.__window = SDL_CreateWindow(windowName, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                         self.__width, self.__height, SDL_WINDOW_SHOWN)
        self.__renderer = SDL_CreateRenderer(self.__window, -1, SDL_RENDERER_ACCELERATED)
        self.__thickness = 7
        self.__isMouseDown = False

    def getMatrix(self: Pixart) -> np.ndarray:
        """
        SUMMARY
            Returns the current matrix in the interface.
        PARAMETERS
            N/A
        RETURN
            np.ndarray: numpy matrix of zeros and ones
        """
        return self.__matrix

    def run(self: Pixart) -> None:
        """
        SUMMARY
            Opens and controls the Pixart window.
        PARAMETERS
            N/A
        RETURN
            None
        """
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

    def __drawGrid(self: Pixart) -> None:
        """
        SUMMARY
            Draws vertical and horizontal lines forming a grid.
        PARAMETERS
            N/A
        RETURN
            None
        """

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

    def __drawFilledSquares(self: Pixart) -> None:
        """
        SUMMARY
            Draws all the filled squares on the canvas.
        PARAMETERS
            N/A
        RETURN
            None
        """
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

    def __squareFinder(self: Pixart, x: int, y: int) -> Tuple[int, int]:
        """
        SUMMARY
            For a mouse pointer position computes which square is under it.
        PARAMETERS
            x int: x coordinate of the mouse
            y int: y coordinate of the mouse
        RETURN
            Tuple[int, int]: the index of the square
        """
        rows, cols = self.__matrix.shape
        i = (x - self.__thickness / 2) / ((self.__width - self.__thickness) / cols)
        j = (y - self.__thickness / 2) / ((self.__height - self.__thickness) / rows)
        return int(i), int(j)

    def __mouseMutateMatrix(self: Pixart, x: int, y: int) -> None:
        """
        SUMMARY
            For a mouse pointer position computes which square is under it and sets
            the underlying matrix accordingly.
        PARAMETERS
            x int: x coordinate of the mouse
            y int: y coordinate of the mouse
        RETURN
            None
        """
        i, j = self.__squareFinder(x, y)
        try:
            self.__matrix[j, i] = 1
        except IndexError:  # the user might be able to select square outside the matrix
            pass


# Runs the pixart window
if __name__ == "__main__":
    pixart = Pixart()
    sys.exit(pixart.run())
