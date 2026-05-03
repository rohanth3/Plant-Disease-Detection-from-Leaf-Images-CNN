import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Container from "@material-ui/core/Container";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import {
  CardActionArea,
  CardMedia,
  Grid,
  Button,
  CircularProgress,
} from "@material-ui/core";
import Clear from "@material-ui/icons/Clear";
import { DropzoneArea } from "material-ui-dropzone";
import { common } from "@material-ui/core/colors";
import backgroundImage from "./background/background.jpg";
import logo from "./logo/logo.jpg";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8001/predict";

const ColorButton = withStyles((theme) => ({
  root: {
    color: "#ffffff",
    backgroundColor: "rgba(77, 153, 0, 0.7)", // matching the appbar green
    backdropFilter: "blur(8px)",
    WebkitBackdropFilter: "blur(8px)",
    border: "1px solid rgba(77, 153, 0, 0.4)",
    boxShadow: "0 4px 30px rgba(0, 0, 0, 0.1)",
    transition: "all 0.3s ease",
    "&:hover": {
      backgroundColor: "rgba(77, 153, 0, 0.9)",
      transform: "translateY(-2px)",
      boxShadow: "0 6px 40px rgba(0, 0, 0, 0.2)",
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  dropZone: {
    backgroundColor: "rgba(255, 255, 255, 0.05) !important",
    border: "2px dashed rgba(255, 255, 255, 0.4) !important",
    minHeight: "300px !important",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    "& .MuiTypography-root": {
      color: "#ffffff !important",
      fontSize: "1.2rem",
      fontWeight: "500",
    },
    "& .MuiSvgIcon-root": {
      color: "#ffffff !important",
    }
  },
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "16px",
    padding: "15px 22px",
    color: "#ffffff",
    fontSize: "20px",
    fontWeight: 900,
    textTransform: "none",
  },
  media: {
    height: 300,
    borderRadius: "16px",
    objectFit: "cover", // prevents image from stretching
  },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  mainContainer: {
    backgroundImage: `url(${backgroundImage})`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundSize: "cover",
    height: "calc(100vh - 64px)",
    width: "100%",
    overflow: "hidden",
  },
  imageCard: {
    margin: "auto",
    [theme.breakpoints.up("md")]: {
      marginLeft: "12vw",
      marginRight: "auto",
    },
    maxWidth: 400,
    height: "auto",
    backgroundColor: "rgba(0, 0, 0, 0.45)", // Dark translucent background
    backdropFilter: "blur(12px)", // Frosted glass effect
    WebkitBackdropFilter: "blur(12px)",
    border: "1px solid rgba(255, 255, 255, 0.2)",
    boxShadow: "0px 8px 32px 0px rgba(0, 0, 0, 0.37) !important",
    borderRadius: "20px",
    overflow: "hidden",
  },
  imageCardEmpty: {
    height: "auto",
  },
  detail: {
    backgroundColor: "transparent",
    display: "flex",
    justifyContent: "center",
    flexDirection: "column",
    alignItems: "center",
    paddingTop: "24px !important",
  },
  appbar: {
    background: "rgba(77, 153, 0, 0.75)", // Sleek translucent green
    backdropFilter: "blur(12px)", // Frosted glass appbar
    WebkitBackdropFilter: "blur(12px)",
    boxShadow: "0 4px 30px rgba(0, 0, 0, 0.1)",
    borderBottom: "1px solid rgba(255, 255, 255, 0.15)",
    color: "white",
  },
  loader: {
    color: "#ffffff !important",
  },
  caption: {
    color: "white",
    fontWeight: "bolder",
    fontSize: "30px",
  },
  loaderText: {
    color: "#ffffff",
    marginTop: "16px",
    fontWeight: "500",
  },
  buttonGrid: {
    maxWidth: "400px",
    width: "100%",
    margin: "auto",
    marginTop: "16px",
    [theme.breakpoints.up("md")]: {
      marginLeft: "12vw",
      marginRight: "auto",
    },
  },
  logoImage: {
    width: 44,
    height: 44,
    borderRadius: "50%",
    marginRight: 16,
    boxShadow: "0px 4px 10px rgba(0,0,0,0.3)",
    border: "2px solid rgba(255, 255, 255, 0.8)",
  },
}));

export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [hasImage, setHasImage] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const confidence = data ? (parseFloat(data.confidence) * 100).toFixed(2) : 0;

  const sendFile = useCallback(async () => {
    if (!hasImage || !selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post(API_URL, formData);
      if (res.status === 200) {
        setData(res.data);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      alert("There was an error processing your image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [hasImage, selectedFile]);

  const clearData = () => {
    setData(null);
    setHasImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  // Generate preview URL whenever a file is selected
  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  // Send to backend once preview is ready
  useEffect(() => {
    if (!preview) return;
    setIsLoading(true);
    sendFile();
  }, [preview, sendFile]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(null);
      setHasImage(false);
      setData(null);
      return;
    }
    setSelectedFile(files[0]);
    setData(null);
    setHasImage(true);
  };

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <img src={logo} alt="Logo" className={classes.logoImage} />
          <Typography className={classes.caption} variant="h6" noWrap>
            Plant Care
          </Typography>
          <div className={classes.grow} />
        </Toolbar>
      </AppBar>

      <Container maxWidth={false} className={classes.mainContainer} disableGutters>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
        >
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!hasImage ? classes.imageCardEmpty : ""}`}>

              {/* Uploaded image preview */}
              {hasImage && (
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={preview}
                    component="img"
                    title="Uploaded plant image"
                  />
                </CardActionArea>
              )}

              {/* Upload dropzone */}
              {!hasImage && (
                <CardContent>
                  <DropzoneArea
                    dropzoneClass={classes.dropZone}
                    acceptedFiles={["image/*"]}
                    dropzoneText={"Click here or Drag and Drop to select an Image"}
                    onChange={onSelectFile}
                  />
                </CardContent>
              )}

              {/* Prediction result */}
              {data && (
                <CardContent className={classes.detail}>
                  <Typography variant="h5" style={{ color: "#ffffff", fontWeight: "bold", textAlign: "center", marginBottom: "4px" }}>
                    {data.class}
                  </Typography>
                  <Typography variant="h6" style={{ color: "#e0e0e0", fontWeight: "400" }}>
                    Confidence: <span style={{ color: "#ffffff", fontWeight: "700" }}>{confidence}%</span>
                  </Typography>
                </CardContent>
              )}

              {/* Loading spinner */}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress color="secondary" className={classes.loader} />
                  <Typography className={classes.loaderText} variant="h6" noWrap>
                    Processing Image...
                  </Typography>
                </CardContent>
              )}

            </Card>
          </Grid>

          {/* Clear button */}
          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton
                variant="contained"
                className={classes.clearButton}
                color="primary"
                component="span"
                size="large"
                onClick={clearData}
                startIcon={<Clear fontSize="large" />}
              >
                Clear
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};
