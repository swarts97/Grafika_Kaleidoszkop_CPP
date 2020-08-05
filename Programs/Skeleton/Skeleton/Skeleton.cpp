//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjeloles kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szoke Tibor Adam
// Neptun : GQ5E7S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Material {
	vec3 ka;
	vec3 kd;
	vec3 ks;
	float shininess;
	vec3 n;
	vec3 kappa;
	bool rough;
	bool reflective;

	Material(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess, bool _rough, bool _refl, vec3 _n = vec3(0, 0, 0), vec3 _kappa = vec3(0, 0, 0)) {
		ka = _ka;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		n = _n;
		kappa = _kappa;
		rough = _rough;
		reflective = _refl;
	}

	vec3 Fresnel(vec3 inDir, vec3 normal) {
		float costheta = dot(inDir, normal);
		vec3 one(1, 1, 1);

		vec3 szamlalo = (n - one) * (n - one) + kappa * kappa;
		vec3 nevezo = (n + one) * (n + one) + kappa * kappa;
		vec3 F0 = vec3(szamlalo.x / nevezo.x, szamlalo.y / nevezo.y, szamlalo.z / nevezo.z);
		return F0 + (one - F0) * pow(1 - costheta, 5);
	}
};

struct Hit {
	float t;
	vec3 position;
	vec3 normal;
	Material * material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start;
	vec3 dir;
	bool out;
	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
};

class Intersectable {
public:
	Material * material;
	virtual Hit intersect(const Ray ray) = 0;
	virtual void move(float Dt) = 0;
};

float rnd() { return (float)rand() / RAND_MAX; }

struct Ellipsoid : public Intersectable {
	float A;
	float B;
	float C;
	vec3 center;
	vec3 moveDir;

	Ellipsoid(float _A, float _B, float _C, vec3 _center, Material* _material) {
		A = _A;
		B = _B;
		C = _C;
		center = _center;
		material = _material;
		moveDir = normalize(vec3(rnd(), rnd(), 0.0f));
	}

	void move(float Dt) {
		vec3 origos = vec3(0.0f, 0.0f, 0.0f);
		vec4 mooveDir = vec4(moveDir.x, moveDir.y, moveDir.z, 1.0f);
		vec3 dist = origos - center;
		if (length(dist) > 0.05f) {
			moveDir = normalize(dist);
		}
		center = center + moveDir * Dt * 0.05;
	}

	Hit intersect(const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - center;

		float a = B * B * C * C * ray.dir.x * ray.dir.x + A * A * C * C * ray.dir.y * ray.dir.y + A * A * B * B * ray.dir.z * ray.dir.z;
		float b = 2 * (B * B * C * C * dist.x * ray.dir.x + A * A * C * C * dist.y * ray.dir.y + A * A * B * B * dist.z * ray.dir.z);
		float c = B * B * C * C * dist.x * dist.x + A * A * C * C * dist.y * dist.y + A * A * B * B * dist.z * dist.z - (A * A * B * B * C * C);

		float discr = (b * b) - 4.0f * a * c;
		if (discr < 0)
			return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / (2.0f * a);
		float t2 = (-b - sqrt_discr) / (2.0f * a);
		if (t1 <= 0)
			return hit;
		hit.t = (t2 > 0) ? t2 : t1;

		hit.position = ray.start + ray.dir * hit.t;
		vec3 normalvec = vec3((hit.position.x - center.x) * 2 / (A * A), (hit.position.y - center.y) * 2 / (B * B), (hit.position.z - center.z) * 2 / (C * C));
		hit.normal = normalize(normalvec);
		hit.material = material;

		return hit;
	}
};

struct Mirror : public Intersectable {
	vec3 a;
	vec3 b;
	vec3 c;
	vec3 d;
	vec3 normalvec;

	Mirror(vec3 _a, vec3 _b, vec3 _c, vec3 _d, Material* _material) {
		a = _a;
		b = _b;
		c = _c;
		d = _d;
		normalvec = normalize(cross(d - a, b - a));
		material = _material;
	}

	void move(float Dt) {}

	Hit intersect(const Ray ray) {
		Hit hit;
		vec3 dist = ray.start - a;
		float t = dot(-dist, normalvec) / dot(ray.dir, normalvec);  
		if (dot(ray.dir, normalvec) != 0.0f) {
			float t = dot(-dist, normalvec) / dot(ray.dir, normalvec);

			vec3 position = ray.start + ray.dir * t;
			if (dot(cross((b - a), (position - a)), normalvec) < 0.0f)
				if (dot(cross((c - b), (position - b)), normalvec) < 0.0f)
					if (dot(cross((d - c), (position - c)), normalvec) < 0.0f)
						if (dot(cross((a - d), (position - d)), normalvec) < 0.0f) {
							hit.t = dot(-dist, normalvec) / dot(ray.dir, normalvec);
							hit.position = ray.start + ray.dir * hit.t;
							hit.normal = normalvec;
							hit.material = material;
							return hit;
						}
		}
		return hit;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir, true);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.000001f;
const int maxdepth = 20;

class Scene {
public:
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
	float mirrorCount = 3;
	float r = 0.1f;
	float bigRound = 2.0f;
	Material * mirrorMat;

	void addMirror() {
		mirrorCount++;
		mirrorBuilder();
	}

	void mirrorBuilder() {
		if (mirrorCount != 3)
			for (int i = 0; i < mirrorCount - 1; i++)
				objects.pop_back();
		vec3 a = vec3(r, 0.0f, 0.0f);
		vec3 b = vec3(r, 0.0f, bigRound);
		for (int i = 0; i < mirrorCount; i++) {
			vec3 c = vec3(cosf((i + 1) * 2 * M_PI / mirrorCount) * r, sinf((i + 1) * 2 * M_PI / mirrorCount) * r, bigRound);
			vec3 d = vec3(cosf((i + 1) * 2 * M_PI / mirrorCount) * r, sinf((i + 1) * 2 * M_PI / mirrorCount) * r, 0.0f);
			objects.push_back(new Mirror(a, b, c, d, mirrorMat));
			a = d;
			b = c;
		}
	}

	void mirrorMatChange(char mat) {
		vec3 kd(0.3f, 0.2f, 0.1f);
		vec3 ks(2, 2, 2);
		vec3 ka = kd * M_PI;
		switch (mat) {
		case 'g': mirrorMat = new Material(ka, kd, ks, 50, false, true, vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f)); break;
		case 's': mirrorMat = new Material(ka, kd, ks, 50, false, true, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f)); break;
		}
		for (int i = 0; i < mirrorCount; i++)
			objects.at(objects.size() - 1 - i)->material = mirrorMat;
	}

	void doAllMoves(float Dt) {
		objects.at(0)->move(Dt);
		objects.at(1)->move(Dt);
		objects.at(2)->move(Dt);
	}

	void build() {
		vec3 eye = vec3(0, 0, 1.5f);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(1.0f, 1.0f, 1.0f);
		vec3 lightDirection(0, 0, 1);
		vec3 Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.15f, 0.1f, 0.05f);
		vec3 ks(0.1f, 0.1f, 0.1f);
		vec3 ka = kd * M_PI;
		Material* arany = new Material(ka, kd, ks, 50, false, true, vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
		Material* ezust = new Material(ka, kd, ks, 50, false, true, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		kd = vec3(0.3f, 0.3f, 0.3f);
		Material* fedlap = new Material(ka, kd, ks, 50, true, false, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		kd = vec3(0.0f, 0.0f, 0.5f);
		Material* kek = new Material(ka, kd, ks, 50, true, false, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		kd = vec3(0.3f, 0.05f, 0.15f);
		Material* piros = new Material(ka, kd, ks, 50, true, false, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		kd = vec3(0.1f, 0.3f, 0.2f);
		Material* zold = new Material(ka, kd, ks, 50, true, false, vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
		mirrorMat = arany;
		
		objects.push_back(new Ellipsoid(0.012f, 0.017f, 0.01f, vec3(-0.02f, -0.025f, 0.0f), piros));
		objects.push_back(new Ellipsoid(0.015f, 0.010f, 0.01f, vec3(-0.01f, 0.02f, 0.0f), kek));
		objects.push_back(new Ellipsoid(0.01f, 0.014f, 0.02f, vec3(0.02f, -0.02f, 0.0f), zold));
		
		objects.push_back(new Mirror(vec3(-0.1f, -0.1f, 0.0f), vec3(0.11f, -0.11f, 0.0f), vec3(0.11f, 0.11f, 0.0f), vec3(-0.11f, 0.11f, 0.0f), fedlap));
		mirrorBuilder();
	}
	
	void render(std::vector<vec4>& image) {
#pragma omp parallel for
		for (int Y = 0; Y < windowHeight; Y++) {
			#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 0);
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
				bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0)
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable * object : objects)
			if (object->intersect(ray).t > 0)
				return true;
		return false;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	}

	vec3 trace(Ray ray, int d = 0) {
		if (d > maxdepth)
			return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
			return La;
		vec3 outRad(0, 0, 0);

		//Direct
		if (hit.material->rough) {
			outRad = hit.material->ka * La;
			for (Light * light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon * 100, light->direction, ray.out);
				Hit shadowHit = firstIntersect(shadowRay);
				if (shadowHit.t < 0) {
					float cosTheta = dot(hit.normal, light->direction);
					if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
						vec3 diffuseRad = light->Le * hit.material->kd * cosTheta;
						outRad = outRad + diffuseRad;
						vec3 halfway = normalize(-ray.dir + light->direction);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0)
							outRad = outRad + light->Le * hit.material->ks * pow(cosDelta, hit.material->shininess);
					}
				}
			}
		}

		//Reflect
		if (hit.material->reflective) {
			vec3 reflectionDir = reflect(ray.dir, hit.normal);
			Ray reflectRay(hit.position + hit.normal * epsilon, reflectionDir, ray.out);
			outRad = outRad + trace(reflectRay, d + 1) * hit.material->Fresnel(-ray.dir, hit.normal);
		}

		return outRad;
	}

};

GPUProgram gpuProgram;
Scene scene;

const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad.Create(image);

	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad.Create(image);

	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	switch (key) {
	case 'a': scene.addMirror(); break;
	case 'g': scene.mirrorMatChange('g'); break;
	case 's': scene.mirrorMatChange('s'); break;
	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	static float tend = 0.0f;
	const float dt = 0.01f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.doAllMoves(Dt);
	}
	glutPostRedisplay();
}