{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creating 1D Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l=[1,2,3]\n",
    "x=np.array(l)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "numpy.ndarray"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2 3\n",
      "(6,)\n",
      "{'name': 'CB'}\n"
     ]
    }
   ],
   "source": [
    "def f(a,b,c=0,d=1,*args, **kwargs):\n",
    "    print(a,b)\n",
    "    print(args)\n",
    "    print(kwargs)\n",
    "\n",
    "    \n",
    "f(2,3,4,5,6,name='CB')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('<U1')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f=np.array(['a','e','i','o','u'],dtype=np.str)\n",
    "f.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "k=np.arange(0,30,2)\n",
    "k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.  ,  1.25,  2.5 ,  3.75,  5.  ,  6.25,  7.5 ,  8.75, 10.  ])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "k=np.linspace(0,10,9)\n",
    "k"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creating 2D Arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [4 5 6]\n",
      " [7 8 9]]\n",
      "(3, 3)\n"
     ]
    }
   ],
   "source": [
    "a = np.array([[1,2,3], [4,5,6], [7,8,9]])\n",
    "print(a)\n",
    "print(a.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.ndim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1.],\n",
       "       [1., 1.],\n",
       "       [1., 1.]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q=np.ones([3,2])\n",
    "q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 1],\n",
       "       [1, 1],\n",
       "       [1, 1]])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q=np.ones([3,2],int)\n",
    "q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0.],\n",
       "       [0., 0.],\n",
       "       [0., 0.]])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros((3,2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 0., 0., 0.],\n",
       "       [0., 1., 0., 0.],\n",
       "       [0., 0., 1., 0.],\n",
       "       [0., 0., 0., 1.]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.eye(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 5, 9])"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.diag(np.array([[1,2,3],[4,5,6],[7,8,9]]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[10, 10, 10],\n",
       "       [10, 10, 10]])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.full((2,3),10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.24335819, 0.70593022, 0.89154462],\n",
       "       [0.43059474, 0.54081189, 0.23202366],\n",
       "       [0.80744999, 0.70305548, 0.56082607]])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "t1=np.random.random((3,3))\n",
    "t1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0, 0],\n",
       "       [0, 2, 0],\n",
       "       [0, 0, 3]])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.diag([1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(9,)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=np.diag([1,2,3]).flatten()\n",
    "x.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reshaping vs Resizing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=np.arange(0,30,2)\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(15,)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  2,  4,  6,  8],\n",
       "       [10, 12, 14, 16, 18],\n",
       "       [20, 22, 24, 26, 28]])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=x.reshape(3,5) #product shd be equal to original array size\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.    1.25  2.5   3.75  5.    6.25  7.5   8.75 10.  ]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 0.  ,  1.25,  2.5 ],\n",
       "       [ 3.75,  5.  ,  6.25],\n",
       "       [ 7.5 ,  8.75, 10.  ]])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "o=np.linspace(0,10,9)\n",
    "print(o)\n",
    "#o=o.resize(3,3) #will not word (inplace)\n",
    "o.resize(3,3)\n",
    "o"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Stacking"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 3, 5],\n",
       "       [2, 4, 6]])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.stack([[1,2],[3,4],[5,6]],axis=1) #along cloumns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4],\n",
       "       [5, 6]])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.vstack?\n",
    "np.vstack([[1,2],[3,4],[5,6]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4, 5, 6])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.hstack?\n",
    "np.hstack([[1,2],[3,4],[5,6]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 5, 6],\n",
       "       [3, 4, 7, 8]])"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1=np.array([[1,2],[3,4]])\n",
    "arr2=np.array([[5,6],[7,8]])\n",
    "np.hstack((arr1,arr2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4]])"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2],\n",
       "       [3, 4],\n",
       "       [5, 6],\n",
       "       [7, 8]])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.vstack((arr1,arr2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Vectorized Operations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=np.array([1,2,3])\n",
    "y=np.array([4,5,6])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x: [1 2 3]\n",
      "y: [4 5 6]\n"
     ]
    }
   ],
   "source": [
    "print(\"x:\",x)\n",
    "print(\"y:\",y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X+Y: [5 7 9]\n",
      "X*Y: [ 4 10 18]\n",
      "X/Y: [0.25 0.4  0.5 ]\n",
      "X**Y: [  1  32 729]\n"
     ]
    }
   ],
   "source": [
    "#pointwise/elementwise\n",
    "print(\"X+Y:\", x+y)\n",
    "print(\"X*Y:\", x*y)\n",
    "print(\"X/Y:\", x/y)\n",
    "print(\"X**Y:\", x**y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[3 6 9] -- array * 3\n",
      "VS\n",
      "[1 2 3 1 2 3 1 2 3] -- list * 3\n",
      "VS\n",
      "[1 1 1 2 2 2 3 3 3] -- repeat method of numpy array\n"
     ]
    }
   ],
   "source": [
    "# Multiplication with a scalar\n",
    "print(np.array([1,2,3]) * 3, \"-- array * 3\")\n",
    "print(\"VS\")\n",
    "print(np.array([1,2,3] * 3), \"-- list * 3\")\n",
    "print(\"VS\")\n",
    "print(np.repeat([1,2,3], 3), \"-- repeat method of numpy array\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "32"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.dot(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 4,  5,  6],\n",
       "       [16, 25, 36]])"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z=np.array([y,y**2])\n",
    "z"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Transposing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 4 16]\n",
      " [ 5 25]\n",
      " [ 6 36]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 4,  5,  6],\n",
       "       [16, 25, 36]])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(z.T)\n",
    "z"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(2, 3) (3, 2)\n"
     ]
    }
   ],
   "source": [
    "print(z.shape,z.T.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6],\n",
       "       [7, 8, 9]])"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l=[[1,2,3],[4,5,6],[7,8,9]]\n",
    "q=np.array(l)\n",
    "q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 2, 2], dtype=int32)"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "q.argmax(axis=1) #returns indices of maximum no along axis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Type casting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 4.  5.  6.]\n",
      " [16. 25. 36.]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "dtype('float32')"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z=z.astype('f')\n",
    "print(z)\n",
    "z.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2D Axis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = np.array([-4,-3,-1,5,19])\n",
    "b = np.array([[1,99,20,31],[-1,-24,37,-13],[-7,11,49,4]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-4 -3 -1  5 19]\n",
      "--------------------\n",
      "[[  1  99  20  31]\n",
      " [ -1 -24  37 -13]\n",
      " [ -7  11  49   4]]\n"
     ]
    }
   ],
   "source": [
    "print(a)\n",
    "print('-'*20)\n",
    "print(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-4 -3 -1  5 19]\n",
      "Max: 19\n",
      "Min: -4\n",
      "Mean: 3.2\n",
      "Std: 8.49470423263812\n",
      "Index of Max Value: 4\n",
      "Max Value 19\n",
      "Index of Min Value: 0\n"
     ]
    }
   ],
   "source": [
    "print(a)\n",
    "print(\"Max:\", a.max())\n",
    "print(\"Min:\", a.min())\n",
    "print(\"Mean:\", a.mean())\n",
    "print(\"Std:\", a.std())\n",
    "print(\"Index of Max Value:\", a.argmax())\n",
    "print(\"Max Value\", a[a.argmax()])\n",
    "print(\"Index of Min Value:\", a.argmin())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(3, 4)"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(12,)"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b.flatten().shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[  1  99  20  31]\n",
      " [ -1 -24  37 -13]\n",
      " [ -7  11  49   4]]\n",
      "\n",
      "Max: 99\n",
      "Min: -24\n",
      "Mean: 17.25\n",
      "Std: 32.00813698629355\n",
      "Index of Max Value: 1\n",
      "Index of Min Value: 5\n",
      "\n",
      "\n",
      "-- ALONG AXIS 0 --\n",
      "Max: [ 1 99 49 31]\n",
      "Min: [ -7 -24  20 -13]\n",
      "Mean: [-2.33333333 28.66666667 35.33333333  7.33333333]\n",
      "Std: [ 3.39934634 51.74510175 11.8977122  18.11690432]\n",
      "Index of Max Value: [0 0 2 0]\n",
      "Index of Min Value: [2 1 0 1]\n",
      "[[  1  99  20  31]\n",
      " [ -1 -24  37 -13]\n",
      " [ -7  11  49   4]]\n",
      "\n",
      "-- ALONG AXIS 1 --\n",
      "Max: [99 37 49]\n",
      "Min: [  1 -24  -7]\n",
      "Mean: [37.75 -0.25 14.25]\n",
      "Std: [36.95520938 22.99320552 21.0638909 ]\n",
      "Index of Max Value: [1 2 2]\n",
      "Index of Min Value: [0 1 0]\n"
     ]
    }
   ],
   "source": [
    "print(b, end='\\n\\n')\n",
    "\n",
    "print(\"Max:\", b.max())\n",
    "print(\"Min:\", b.min())\n",
    "print(\"Mean:\", b.mean())\n",
    "print(\"Std:\", b.std())\n",
    "print(\"Index of Max Value:\", b.argmax()) # Returns position of the flattened\n",
    "print(\"Index of Min Value:\", b.argmin())\n",
    "\n",
    "print('\\n\\n-- ALONG AXIS 0 --')\n",
    "print(\"Max:\", b.max(axis=0))\n",
    "print(\"Min:\", b.min(axis=0))\n",
    "print(\"Mean:\", b.mean(axis=0))\n",
    "print(\"Std:\", b.std(axis=0))\n",
    "print(\"Index of Max Value:\", b.argmax(axis=0)) # Returns position of the flattened\n",
    "print(\"Index of Min Value:\", b.argmin(axis=0))\n",
    "print(b)\n",
    "print('\\n-- ALONG AXIS 1 --')\n",
    "print(\"Max:\", b.max(axis=1))\n",
    "print(\"Min:\", b.min(axis=1))\n",
    "print(\"Mean:\", b.mean(axis=1))\n",
    "print(\"Std:\", b.std(axis=1))\n",
    "print(\"Index of Max Value:\", b.argmax(axis=1)) # Returns position of the flattened\n",
    "print(\"Index of Min Value:\", b.argmin(axis=1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Indexing and Slicing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144],\n",
       "      dtype=int32)"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#a=np.arange(0,13,1)\n",
    "#a=a**2\n",
    "a=np.arange(13)**2\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0, 16, array([0, 4], dtype=int32))"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[0],a[4],a[0:3:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([144, 121, 100,  81,  64,  49,  36,  25,  16,   9,   4,   1,   0],\n",
       "      dtype=int32)"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[::-1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 81, 100, 121, 144], dtype=int32)"
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[-4:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144],\n",
       "      dtype=int32)"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([64, 36, 16,  4], dtype=int32)"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a[-5:0:-2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  3,  4,  5],\n",
       "       [ 6,  7,  8,  9, 10, 11],\n",
       "       [12, 13, 14, 15, 16, 17],\n",
       "       [18, 19, 20, 21, 22, 23],\n",
       "       [24, 25, 26, 27, 28, 29],\n",
       "       [30, 31, 32, 33, 34, 35]])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r = np.arange(36)\n",
    "r.resize((6,6))\n",
    "r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "14\n",
      "14\n",
      "[12 13 14 15 16 17]\n"
     ]
    }
   ],
   "source": [
    "print(r[2,2]) # Second row, second column\n",
    "print(r[2][2])\n",
    "print(r[2])\n",
    "#Notice the ',' usage while indexing as compared to [][] usage to access multidimensional lists"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[22, 23],\n",
       "       [28, 29]])"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r[3:5,-2:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([35, 33, 31])"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r[-1,::-2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Coditional Indexing(Index Arrays)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False, False, False, False, False, False],\n",
       "       [False,  True,  True,  True,  True,  True]])"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "verdict=r>30\n",
    "verdict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([31, 32, 33, 34, 35])"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "greater_than_30 = r[verdict]\n",
    "greater_than_30"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1  2  3  4  5]\n",
      " [ 6  7  8  9 10 11]\n",
      " [12 13 14 15 16 17]\n",
      " [18 19 20 21 22 23]\n",
      " [24 25 26 27 28 29]\n",
      " [30 31 32 33 34 35]]\n",
      "[[False False False False False False]\n",
      " [False False False False False False]\n",
      " [False False False False False False]\n",
      " [False False False False False False]\n",
      " [False False False False False False]\n",
      " [False  True  True  True  True  True]]\n",
      "5\n",
      "165\n"
     ]
    }
   ],
   "source": [
    "print(r)\n",
    "print(verdict)\n",
    "print((r>30).sum())\n",
    "print((r[r>30]).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  3,  4,  5],\n",
       "       [ 6,  7,  8,  9, 10, 11],\n",
       "       [12, 13, 14, 15, 16, 17],\n",
       "       [18, 19, 20, 21, 22, 23],\n",
       "       [24, 25, 26, 27, 28, 29],\n",
       "       [30,  9,  9,  9,  9,  9]])"
      ]
     },
     "execution_count": 95,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r[r>30]=9\n",
    "r"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2],\n",
       "       [ 6,  7,  8],\n",
       "       [12, 13, 14]])"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2=r[:3,:3]\n",
    "r2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "r2[0,0]=100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[100,   1,   2],\n",
       "       [  6,   7,   8],\n",
       "       [ 12,  13,  14]])"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "  C_CONTIGUOUS : True\n",
       "  F_CONTIGUOUS : False\n",
       "  OWNDATA : True\n",
       "  WRITEABLE : True\n",
       "  ALIGNED : True\n",
       "  WRITEBACKIFCOPY : False\n",
       "  UPDATEIFCOPY : False"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r.flags #****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# += vs ="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 3 4 5]\n",
      "[2 3 4 5]\n"
     ]
    }
   ],
   "source": [
    "a=np.array([1,2,3,4])\n",
    "b=a\n",
    "a+=np.array([1,1,1,1]) #inplace\n",
    "print(a)\n",
    "print(b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a: [2 3 4 5]\n",
      "b: [1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,4])\n",
    "b = a\n",
    "a = a + np.array([1,1,1,1]) #not inplace\n",
    "print('a:', a)\n",
    "print('b:', b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Copying"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[1, 9, 1],\n",
       "       [3, 9, 8],\n",
       "       [7, 3, 7],\n",
       "       [3, 6, 3]])"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import random\n",
    "print(random.randint(0,10))\n",
    "test = np.random.randint(0, 10, (4,3))\n",
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 9 1]\n",
      "[3 9 8]\n",
      "[7 3 7]\n",
      "[3 6 3]\n"
     ]
    }
   ],
   "source": [
    "for row in test:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Row 0 is [1 9 1]\n",
      "Row 1 is [3 9 8]\n",
      "Row 2 is [7 3 7]\n",
      "Row 3 is [3 6 3]\n"
     ]
    }
   ],
   "source": [
    "for index in range(len(test)):\n",
    "    print(\"Row {} is {}\".format(index, test[index]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Row 0 is [1 9 1]\n",
      "Row 1 is [3 9 8]\n",
      "Row 2 is [7 3 7]\n",
      "Row 3 is [3 6 3]\n"
     ]
    }
   ],
   "source": [
    "for i, row in enumerate(test):\n",
    "        print(\"Row {} is {}\".format(i, row))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1, 81,  1],\n",
       "       [ 9, 81, 64],\n",
       "       [49,  9, 49],\n",
       "       [ 9, 36,  9]], dtype=int32)"
      ]
     },
     "execution_count": 110,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test2 = test**2\n",
    "test2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 9 1] + [ 1 81  1] = [ 2 90  2]\n",
      "[3 9 8] + [ 9 81 64] = [12 90 72]\n",
      "[7 3 7] + [49  9 49] = [56 12 56]\n",
      "[3 6 3] + [ 9 36  9] = [12 42 12]\n"
     ]
    }
   ],
   "source": [
    "for i,j in zip(test, test2):\n",
    "    print(i, '+', j, '=', i+j)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(array([1, 9, 1]), array([ 1, 81,  1], dtype=int32)),\n",
       " (array([3, 9, 8]), array([ 9, 81, 64], dtype=int32)),\n",
       " (array([7, 3, 7]), array([49,  9, 49], dtype=int32)),\n",
       " (array([3, 6, 3]), array([ 9, 36,  9], dtype=int32))]"
      ]
     },
     "execution_count": 112,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(zip(test, test2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "'list' object is not callable",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-113-ad2d0b9634ae>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mlist\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m: 'list' object is not callable"
     ]
    }
   ],
   "source": [
    "list = list()\n",
    "list() #made list as object"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[]\n"
     ]
    }
   ],
   "source": [
    "print(list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
