#version 330 core
out vec4 FragColor;

// Informații primite din vertex shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec4 FragPosLightSpace;

// Structură pentru lumina punctuală
struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

// Structură pentru material
struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

uniform PointLight pointLight;
uniform Material material;
uniform vec3 viewPos; // poziția camerei
uniform sampler2D shadowMap;

// ------------------------------------------------
// Calcularea umbrei (ShadowCalculation) în perspectivă
// ------------------------------------------------
float ShadowCalculation(vec4 fragPosLightSpace, vec3 normal)
{
    // Convertim coordonatele [-1,1] -> [0,1]
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    // Verificăm dacă fragmentul e în afara "viziunii" luminii
    if(projCoords.z > 1.0) {
        return 0.0;
    }

    // Luăm adâncimea cea mai apropiată din textura de adâncime
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    // Adâncimea fragmentului curent (în [0,1])
    float currentDepth = projCoords.z;

    // Bias pentru a evita efectul de "shadow acne"
    // poate fi adaptat în funcție de scenă
    float bias = max(0.05 * (1.0 - dot(normal, normalize(pointLight.position - FragPos))), 0.005);

    // PCF (Percentage-Closer Filtering) – 3x3 e.g.
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; x++) {
        for(int y = -1; y <= 1; y++) {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x,y) * texelSize).r; 
            shadow += (currentDepth - bias > pcfDepth) ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    return shadow;
}

void main()
{
    // =====================
    // 1) Calcul lumina punctuală
    // =====================
    vec3 norm    = normalize(Normal);
    vec3 lightDir = normalize(pointLight.position - FragPos);

    // Ambient
    vec3 ambient = pointLight.ambient * texture(material.diffuse, TexCoord).rgb;

    // Diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = pointLight.diffuse * diff * texture(material.diffuse, TexCoord).rgb;

    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    vec3 specular = pointLight.specular * spec * texture(material.specular, TexCoord).rgb;

    // =====================
    // 2) Calcul umbră
    // =====================
    float shadow = ShadowCalculation(FragPosLightSpace, norm);

    // =====================
    // 3) Combinație finală
    // =====================
    vec3 lighting = ambient + (1.0 - shadow) * (diffuse + specular);
    FragColor = vec4(lighting, 1.0);
}
