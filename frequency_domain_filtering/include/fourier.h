/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-26 11:23:12
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-26 11:23:12
 * @Description: from the start, we will learn the frequency domain filtering what
 * can be also named as the fourier transform.
 * what is fourier transform? any periodic function can be expressed as different sine or
 * cosine and form, and each sine and cosine multipied by different coefficient. it is 
 * fourier series. 
 * any aperiodic function can be expressed as sine or cosine and multipied by the
 * weighting function integral. it is fourier transform.
 * notice, in some specific areas, the fourier transform is greater than the fourier series.
 * because we can use fourier inverse transform to return the original domain. and 
 * we can do something in frequency domain and will not lose any information.
 * 
 * before we start, we should learn some basic knowledge.
 * complex number: it can be expressed used a + bi, a is the real component, 
 * b is imaginary part. i is imaginary part unit, it meet i^2 = -1.
 * complex number can express the vector on the plane. and support
 * add, sub, multi and divide operation. the plural of complex number is to take a real
 * constant and take the opposite opeartion. it means the plural of the complex a+bi is equal
 * to a-bi.
 * the complex number can be as one point in one cartesian coordinate system.
 * just like one complex C = R+jI, it is the point (R, I) in cartesian coordinate system.
 * sometimes, we can represent the complex number in polar coordinates.
 * C = |C|(cosθ+j*sinθ)， |C| = (R^2+T^2)^(1/2) is the length from the original point in cartesian coordinate
 * to the point (R, I).
 * tanθ = y/x
 * arctan y/x = θ
 * r = (x^2+y^2)^(1/2)
 * 
 * then, we can represent one complex used rectangular coordinate system and plural coordinate.
 * just like one complex number C = x+yI
 * it means one point (x, y) in rectangular coordinate system.
 * you can also transform the complex C based on the plural coordinate.
 * you can use euler's formula.
 * C = r(cosθ+i*sinθ) = (x^2+y^2)^(1/2) * (cosθ + i*sinθ)
 * why?
 * 
 * just like one complex number C = x+yi, it mean the point(x, y) in rectangular coordinate.
 * and (r, θ) in plural coordinate. r is the length from original point to the point(x, y)
 * in plural coordinate system.
 * for plural coordinate, x = r * cosθ, y = r * sinθ, θ = arctan y/x
 * C = x+yi = r*cosθ+(r*sinθ)i = r(cosθ+sinθ*i) = r*e^(iθ)
 * one standard eulaer's formula is e^iθ = cosθ+sinθ*i
 * so the eular's formular in plural coordinate is e^iθ = cosθ+sinθ*i.
 * 
 * so C = x+yI
 * this complex can be expressed as C = |C|*e^iθ = |C|*(cosθ+sinθ*I)
 * |C| = (x^2+y^2)^(1/2) = r
 * I = (-1)^(1/2)
 * the value arctan is range from -π/2 to π/2
 * this is complex number above, we can also define the complex function used the same method.
 * just like one complex number C = 1 + j2, j is equal to (-1)^(1/2)
 * it is the point (1, 2) in rectangular coordinate system.
 * the expression for plural coordinate is C = r*e^jθ = (1^2+2^2)^(1/2)*e^(jθ) = (5)^(1/2)*e^(jθ)
 * θ = arctan 2 = 64.4 degree or 1.1 radian.
 * you can use numpy.arctan(2) or math.atan(2). it will return the radian.
 * radian = degree/360*2πr
 * degree = radian/2πr*360 = 1.1/[2*3.14*1]*360 = 63
 * notice the standard radina is calculated based on the r 1.
 * you can also calculate the degree based on the radian used numpy.degrees(radian) function.
 * 
 * just like one complex function
 * F(u) = R(u) + jI(u)
 * R(u) is real volume function, I(u) is virtual volume function.
 * F(u)* = R(u) - jI(u)
 * |F(u)| = [R(u)^2 + I(u)^2]^(1/2)
 * θ(u) = arctan[I(u)/R(u)]
 * 
 * impulse:
 *      the impulse expression of continuous varible t in t=0
 *      δ(t) = ∞, t = 0; 0, t ≠ 0；
 *      the limit condition is the integral from -∞ to +∞ of δ(t) is equal to 1.
 *      ∫(-∞, +∞)δ(t)dt=1
 * on the physical, if t is time, one impulse can be as a peak signal
 * that infinite amplitude and the duration is zero.
 * one impulse has the sifting feature as follow.
 * ∫(-∞, +∞)f(t)δ(t)dt = f(0)
 *  
***********************************************************************/
#include "general.h"


