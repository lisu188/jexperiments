FROM eclipse-temurin:17-jdk AS build

WORKDIR /workspace
COPY . .
RUN ./gradlew --no-daemon :blogsite:bootJar

FROM eclipse-temurin:17-jre

WORKDIR /app
COPY --from=build /workspace/blogsite/build/libs/ExperimentBlogSite-*.jar /app/app.jar

EXPOSE 8080
ENTRYPOINT ["sh", "-c", "exec java ${JAVA_OPTS:-} -jar /app/app.jar --server.port=${PORT:-8080}"]
