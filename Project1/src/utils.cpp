
#include "utils.h"

// hsv ת rgb
Vec3f HSVtoRGB(Vec3f hsv)
{
	int i;
	float f, p, q, t;
	if (hsv[1] == 0) {

		// achromatic (grey)
		return Vec3f(hsv[2], hsv[2], hsv[2]);;
	}
	hsv[0] /= 60;            // sector 0 to 5
	i = floor(hsv[0]);
	f = hsv[0] - i;          // factorial part of h
	p = hsv[2] * (1 - hsv[1]);
	q = hsv[2] * (1 - hsv[1] * f);
	t = hsv[2] * (1 - hsv[1] * (1 - f));
	switch (i) 
	{
	case 0:
		return Vec3f(hsv[2], t, p);
	case 1:
		return Vec3f(q, hsv[2], p);
	case 2:
		return Vec3f(p, hsv[2], t);
	case 3:
		return Vec3f(p, q, hsv[2]);
	case 4:
		return Vec3f(t, p, hsv[2]);
	default:		// case 5:
		return Vec3f(hsv[2], p, q);
	}
}
